
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# PROJECT CONFIGURATION
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SCRIPT = ROOT / "src" / "32_simulate_relay_deployment.py"
OUTPUT_DIR = ROOT / "data" / "processed" / "script35"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RADIO_RANGES = [8.0, 10.0, 12.0]
PLACEMENT_STRATEGIES = ["current_position", "movement_aware"]
POLICIES = ["reactive", "fixed_lead", "predictive"]

PROBABILITY_CUTOFF = 0.70
COOLDOWN_SEC = 30.0
LEAD_BAND = 10.0
PREDICTION_WINDOW_SEC = 30.0
OBSERVATION_HORIZON_SEC = 120.0

VELOCITY_HISTORY_SEC = 5.0

# Load the original simulator without modifying it.
spec = importlib.util.spec_from_file_location(
    "simulator32", SOURCE_SCRIPT
)
simulator32 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(simulator32)


# ---------------------------------------------------------
# MOVEMENT-AWARE POSITION ESTIMATION
# ---------------------------------------------------------

def estimate_future_position(times, xy, index, horizon_sec):
    """
    Estimate the explorer position at relay arrival time.

    Uses only observations at or before the dispatch timestamp.
    Fits a straight line independently to x and y over the
    recent history window. This is a simple baseline predictor,
    not a trained motion-planning model.
    """
    current_time = float(times[index])
    current_position = np.asarray(xy[index], dtype=float)

    if horizon_sec <= 0:
        return tuple(current_position)

    start_time = current_time - VELOCITY_HISTORY_SEC

    # Past and present samples only: no future-position leakage.
    start_index = np.searchsorted(
        times[:index + 1], start_time, side="left"
    )

    history_times = times[start_index:index + 1]
    history_xy = xy[start_index:index + 1]

    if len(history_times) < 2:
        return tuple(current_position)

    relative_times = history_times - current_time

    # Fit x(t) and y(t) using only the available history.
    vx = np.polyfit(relative_times, history_xy[:, 0], 1)[0]
    vy = np.polyfit(relative_times, history_xy[:, 1], 1)[0]

    predicted = current_position + np.array([vx, vy]) * horizon_sec

    if not np.all(np.isfinite(predicted)):
        return tuple(current_position)

    return float(predicted[0]), float(predicted[1])


# ---------------------------------------------------------
# SIMULATION
# ---------------------------------------------------------

def simulate(
    df,
    policy,
    placement_strategy,
    radio_range,
    base,
):
    df = (
        df.sort_values("elapsed_sec")
        .drop_duplicates("elapsed_sec", keep="first")
        .reset_index(drop=True)
    )

    times = df["elapsed_sec"].to_numpy(dtype=float)
    signal = df["signal_quality_smooth"].to_numpy(dtype=float)
    threshold = float(df["signal_threshold"].iloc[0])

    deployment = float(df["deployment_time_sec"].iloc[0])
    reconnect = float(df["reconnect_time_sec"].iloc[0])
    placement_distance = float(df["dispatch_distance_m"].iloc[0])
    margin = float(df["safety_margin_sec"].iloc[0])

    xy = df[["robot_pos_x", "robot_pos_y"]].to_numpy(dtype=float)
    changes = df["signal_5s_change"].fillna(0).to_numpy(dtype=float)
    probs = df["crossing_probability_120s"].fillna(0).to_numpy(dtype=float)
    predicted_times = df[
        "predicted_time_to_crossing_sec"
    ].to_numpy(dtype=float)

    crossing_indices = simulator32.find_crossings(
        times, signal, threshold
    )
    crossing_times = times[crossing_indices]

    nodes = [base]
    connected = np.zeros(len(times), dtype=bool)

    dispatches = []
    pending = None
    last_dispatch_time = -np.inf

    for i, t in enumerate(times):
        explorer = (float(xy[i, 0]), float(xy[i, 1]))

        # Complete a pending deployment when its arrival is observed.
        if pending is not None and pending["arrival_time"] <= t:
            parent_index = pending["parent_index"]
            target = pending["target"]
            parent = nodes[parent_index]

            if (
                parent_index == len(nodes) - 1
                and simulator32.distance(parent, target) <= radio_range
            ):
                nodes.append(target)
                pending["log"]["deployment_status"] = "deployed"
            else:
                pending["log"]["deployment_status"] = (
                    "placement_link_out_of_range"
                )

            pending = None

        # Evaluate connectivity using the actual position at this sample.
        connected[i] = simulator32.chain_is_connected(
            nodes, explorer, radio_range
        )

        if pending is not None:
            continue

        if t - last_dispatch_time < COOLDOWN_SEC:
            continue

        # Keep the original Script 32 dispatch rules unchanged.
        if policy == "reactive":
            trigger = (
                i > 0
                and signal[i - 1] > threshold
                and signal[i] <= threshold
            )

        elif policy == "fixed_lead":
            trigger = (
                signal[i] > threshold
                and signal[i] <= threshold + LEAD_BAND
                and changes[i] < 0
            )

        elif policy == "predictive":
            predicted = predicted_times[i]
            trigger = (
                signal[i] > threshold
                and probs[i] >= PROBABILITY_CUTOFF
                and np.isfinite(predicted)
                and deployment + margin
                <= predicted
                <= deployment + margin + PREDICTION_WINDOW_SEC
            )

        elif policy == "adaptive":
            predicted = predicted_times[i]
            operational_delay = deployment
            trigger = (
                signal[i] > threshold
                and probs[i] >= PROBABILITY_CUTOFF
                and np.isfinite(predicted)
                and 0 <= predicted
                <= operational_delay + margin
            )

        else:
            raise ValueError(f"Unknown policy: {policy}")

        if not trigger:
            continue

        # deployment_time_sec already includes travel, setup and reconnect.
        arrival_delay = deployment

        # Strategy A uses the current position.
        if placement_strategy == "current_position":
            placement_reference = explorer

        # Strategy B estimates the position at usable relay arrival.
        elif placement_strategy == "movement_aware":
            placement_reference = estimate_future_position(
                times, xy, i, arrival_delay
            )

        else:
            raise ValueError(
                f"Unknown placement strategy: {placement_strategy}"
            )

        target = simulator32.next_relay_position(
            nodes[-1], placement_reference, placement_distance
        )

        if target is None:
            continue

        arrival_time = float(t + arrival_delay)

        log = {
            "recording_id": df["recording_id"].iloc[0],
            "signal_threshold": threshold,
            "radio_range_m": radio_range,
            "policy": policy,
            "placement_strategy": placement_strategy,
            "dispatch_time_sec": float(t),
            "arrival_time_sec": arrival_time,
            "deployment_time_sec": deployment,
            "reconnect_time_sec": reconnect,
            "predicted_crossing_probability": float(probs[i]),
            "predicted_time_to_crossing_sec": (
                float(predicted_times[i])
                if np.isfinite(predicted_times[i])
                else np.nan
            ),
            "dispatch_x": explorer[0],
            "dispatch_y": explorer[1],
            "placement_reference_x": placement_reference[0],
            "placement_reference_y": placement_reference[1],
            "target_x": target[0],
            "target_y": target[1],
            "deployment_status": "pending_at_recording_end",
            "crossing_outcome": "",
        }

        dispatches.append(log)

        pending = {
            "arrival_time": arrival_time,
            "target": target,
            "parent_index": len(nodes) - 1,
            "log": log,
        }

        last_dispatch_time = t

    if pending is not None:
        pending["log"]["deployment_status"] = "arrival_not_observed"

    # Match Script 32's crossing-based dispatch diagnostic.
    for d in dispatches:
        t = d["dispatch_time_sec"]
        j = int(np.searchsorted(crossing_times, t, side="left"))

        if j < len(crossing_times):
            lead = crossing_times[j] - t
        else:
            lead = np.inf

        if lead <= OBSERVATION_HORIZON_SEC:
            if d["arrival_time_sec"] <= crossing_times[j]:
                d["crossing_outcome"] = "arrival_before_crossing"
            else:
                d["crossing_outcome"] = "arrival_after_crossing"
        elif times[-1] - t >= OBSERVATION_HORIZON_SEC:
            d["crossing_outcome"] = "false_alarm_complete_window"
        else:
            d["crossing_outcome"] = "censored_observation_window"

    # Time-weighted connectivity and outage calculations.
    if len(times) >= 2:
        dt = np.diff(times)
        valid_dt = dt > 0

        total_time = float(dt[valid_dt].sum())
        connected_time = float(
            dt[valid_dt & connected[:-1]].sum()
        )
        outage_seconds = float(
            dt[valid_dt & ~connected[:-1]].sum()
        )

        outage_starts = int(
            np.sum((~connected[1:]) & connected[:-1])
        )
        outage_starts += int(not connected[0])
    else:
        total_time = connected_time = outage_seconds = 0.0
        outage_starts = int(not connected[0]) if len(times) else 0

    crossing_connected = int(
        sum(connected[i] for i in crossing_indices)
    )

    return {
        "recording_id": df["recording_id"].iloc[0],
        "signal_threshold": threshold,
        "radio_range_m": radio_range,
        "policy": policy,
        "placement_strategy": placement_strategy,
        "relay_speed_mps": float(df["relay_speed_mps"].iloc[0]),
        "deployment_time_sec": deployment,
        "reconnect_time_sec": reconnect,
        "dispatch_distance_m": placement_distance,
        "observed_crossing_events": len(crossing_indices),
        "connected_at_crossing": crossing_connected,
        "crossing_connectivity_pct": (
            100 * crossing_connected / len(crossing_indices)
            if crossing_indices else np.nan
        ),
        "total_observation_sec": total_time,
        "connected_time_sec": connected_time,
        "connectivity_availability_pct": (
            100 * connected_time / total_time
            if total_time > 0 else np.nan
        ),
        "simulated_outage_sec": outage_seconds,
        "simulated_outage_starts": outage_starts,
        "total_dispatches": len(dispatches),
        "relays_deployed": sum(
            d["deployment_status"] == "deployed"
            for d in dispatches
        ),
        "placement_failures": sum(
            d["deployment_status"] == "placement_link_out_of_range"
            for d in dispatches
        ),
        "arrivals_not_observed": sum(
            d["deployment_status"] == "arrival_not_observed"
            for d in dispatches
        ),
        "arrival_before_crossing": sum(
            d["crossing_outcome"] == "arrival_before_crossing"
            for d in dispatches
        ),
        "arrival_after_crossing": sum(
            d["crossing_outcome"] == "arrival_after_crossing"
            for d in dispatches
        ),
        "false_alarms_complete_window": sum(
            d["crossing_outcome"] == "false_alarm_complete_window"
            for d in dispatches
        ),
        "censored_dispatches": sum(
            d["crossing_outcome"] == "censored_observation_window"
            for d in dispatches
        ),
        "dispatch_log": dispatches,
    }


# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------

def main():
    print("Loading data using Script 32...")
    data = simulator32.load_data()

    # Same trajectories/scenarios are evaluated with both strategies.
    grouped = data.groupby(
        simulator32.SCENARIO_KEYS,
        sort=False,
        dropna=False,
    )

    total_groups = len(grouped)
    detail_rows = []
    dispatch_rows = []

    total_runs = (
        total_groups
        * len(RADIO_RANGES)
        * len(POLICIES)
        * len(PLACEMENT_STRATEGIES)
    )
    run_count = 0

    for _, trajectory in grouped:
        for radio_range in RADIO_RANGES:
            for policy in POLICIES:
                for strategy in PLACEMENT_STRATEGIES:
                    result = simulate(
                        trajectory,
                        policy=policy,
                        placement_strategy=strategy,
                        radio_range=radio_range,
                        base=(0.0, 0.0),
                    )

                    dispatch_log = result.pop("dispatch_log")
                    detail_rows.append(result)
                    dispatch_rows.extend(dispatch_log)

                    run_count += 1

        if run_count % 100 == 0:
            print(f"Completed {run_count}/{total_runs} simulation runs")

    details = pd.DataFrame(detail_rows)
    dispatches = pd.DataFrame(dispatch_rows)

    details.to_csv(
        OUTPUT_DIR / "recording_policy_details.csv", index=False
    )
    dispatches.to_csv(
        OUTPUT_DIR / "dispatch_log.csv", index=False
    )

    # Pool counts and denominators rather than averaging percentages.
    summary_rows = []

    group_columns = [
        "radio_range_m", "policy", "placement_strategy"
    ]

    for keys, group in details.groupby(group_columns, dropna=False):
        radio_range, policy, strategy = keys

        crossings = int(group["observed_crossing_events"].sum())
        connected_crossings = int(group["connected_at_crossing"].sum())
        total_time = float(group["total_observation_sec"].sum())
        connected_time = float(group["connected_time_sec"].sum())
        total_dispatches = int(group["total_dispatches"].sum())
        deployed = int(group["relays_deployed"].sum())

        summary_rows.append({
            "radio_range_m": radio_range,
            "policy": policy,
            "placement_strategy": strategy,
            "trajectories": len(group),
            "observed_crossing_events": crossings,
            "connected_at_crossing": connected_crossings,
            "crossing_connectivity_pct": (
                100 * connected_crossings / crossings
                if crossings else np.nan
            ),
            "connectivity_availability_pct": (
                100 * connected_time / total_time
                if total_time else np.nan
            ),
            "simulated_outage_sec": group["simulated_outage_sec"].sum(),
            "simulated_outage_starts": group["simulated_outage_starts"].sum(),
            "total_dispatches": total_dispatches,
            "relays_deployed": deployed,
            "deployed_per_dispatch_pct": (
                100 * deployed / total_dispatches
                if total_dispatches else np.nan
            ),
            "placement_failures": group["placement_failures"].sum(),
            "arrivals_not_observed": group["arrivals_not_observed"].sum(),
            "arrival_before_crossing": group["arrival_before_crossing"].sum(),
            "arrival_after_crossing": group["arrival_after_crossing"].sum(),
            "false_alarms_complete_window": (
                group["false_alarms_complete_window"].sum()
            ),
            "censored_dispatches": group["censored_dispatches"].sum(),
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "policy_summary.csv", index=False)

    # Direct paired comparison: movement-aware minus current-position.
    index_columns = ["radio_range_m", "policy"]

    current = summary[
        summary["placement_strategy"] == "current_position"
    ].set_index(index_columns)

    aware = summary[
        summary["placement_strategy"] == "movement_aware"
    ].set_index(index_columns)

    comparison_metrics = [
        "crossing_connectivity_pct",
        "connectivity_availability_pct",
        "simulated_outage_sec",
        "total_dispatches",
        "placement_failures",
        "false_alarms_complete_window",
    ]

    comparison = pd.DataFrame(index=current.index).reset_index()

    for metric in comparison_metrics:
        comparison[f"current_{metric}"] = current[metric].to_numpy()
        comparison[f"movement_aware_{metric}"] = aware[metric].to_numpy()
        comparison[f"change_{metric}"] = (
            aware[metric].to_numpy() - current[metric].to_numpy()
        )

    comparison.to_csv(
        OUTPUT_DIR / "placement_strategy_comparison.csv",
        index=False,
    )

    print("\nPOOLED RESULTS")
    print(summary.round(2).to_string(index=False))

    print("\nMOVEMENT-AWARE MINUS CURRENT-POSITION")
    print(comparison.round(2).to_string(index=False))

    print(f"\nSaved outputs to: {OUTPUT_DIR}")
    print(
        "WARNING: Connectivity is a simplified geometric simulation, "
        "not measured radio performance."
    )


if __name__ == "__main__":
    main()
