
from pathlib import Path
import argparse
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "data" / "processed" / "script24"
PREDICTION_FILE = (
    ROOT / "data" / "processed" / "script29"
    / "two_stage_oof_predictions.csv"
)
OUTPUT_DIR = ROOT / "data" / "processed" / "script32"

SCENARIO_KEYS = [
    "recording_id",
    "signal_threshold",
    "relay_speed_mps",
    "setup_time_sec",
    "reconnect_time_sec",
    "dispatch_distance_m",
    "safety_margin_sec",
]

REQUIRED_PREDICTION_COLUMNS = [
    "recording_id",
    "elapsed_sec",
    "signal_quality",
    "signal_5s_change",
    "signal_threshold",
    "deployment_time_sec",
    "reconnect_time_sec",
    "dispatch_distance_m",
    "safety_margin_sec",
    "crossing_probability_120s",
    "predicted_time_to_crossing_sec",
]

POLICIES = ["reactive", "fixed_lead", "predictive"]


def load_data():
    if not PREDICTION_FILE.exists():
        raise FileNotFoundError(
            f"Missing predictions file: {PREDICTION_FILE}"
        )

    inputs_file = INPUT_DIR / "simulation_inputs.csv"
    if not inputs_file.exists():
        raise FileNotFoundError(
            f"Missing simulation inputs: {inputs_file}"
        )

    pred = pd.read_csv(PREDICTION_FILE)
    positions = pd.read_csv(inputs_file)

    missing = set(REQUIRED_PREDICTION_COLUMNS) - set(pred.columns)
    if missing:
        raise ValueError(
            f"Prediction file is missing columns: {sorted(missing)}"
        )

    position_cols = [
        "recording_id",
        "elapsed_sec",
        "robot_pos_x",
        "robot_pos_y",
    ]
    missing = set(position_cols) - set(positions.columns)
    if missing:
        raise ValueError(
            f"Simulation inputs are missing columns: {sorted(missing)}"
        )

    # One coordinate record per recording/time.
    # Preserve original timestamps; round only a temporary join key.
    positions = positions[position_cols].copy()
    positions = positions.dropna(
        subset=["recording_id", "elapsed_sec",
                "robot_pos_x", "robot_pos_y"]
    )
    positions = positions.drop_duplicates(
        ["recording_id", "elapsed_sec"], keep="first"
    )

    pred["_join_time"] = pred["elapsed_sec"].round(9)
    positions["_join_time"] = positions["elapsed_sec"].round(9)
    join_keys = ["recording_id", "_join_time"]

    if positions.duplicated(join_keys).any():
        raise ValueError(
            "Rounded timestamp collision in coordinates; "
            "cannot safely join prediction and position data."
        )

    # Keep prediction elapsed_sec as the authoritative simulation time.
    positions = positions.drop(columns=["elapsed_sec"])

    pred = pred.merge(
        positions,
        on=join_keys,
        how="left",
        validate="many_to_one",
    )
    pred = pred.drop(columns=["_join_time"])

    pred = pred.dropna(
        subset=[
            "recording_id", "elapsed_sec",
            "robot_pos_x", "robot_pos_y", "signal_quality"
        ]
    )

    pred = pred.sort_values(
        SCENARIO_KEYS + ["elapsed_sec"]
    ).reset_index(drop=True)

    if pred.empty:
        raise ValueError("No usable prediction/trajectory rows remain.")

    print(f"Prediction rows with coordinates: {len(pred):,}")
    print(f"Recordings: {pred['recording_id'].nunique()}")
    print(f"Scenario trajectories: {len(pred.groupby(SCENARIO_KEYS))}")

    return pred


def distance(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def next_relay_position(parent, explorer, placement_distance):
    """Place a relay toward the explorer from the latest relay."""
    dx = explorer[0] - parent[0]
    dy = explorer[1] - parent[1]
    dist = float(np.hypot(dx, dy))

    if dist < 1e-9:
        return None

    # Do not place the relay beyond the explorer.
    step = min(float(placement_distance), 0.8 * dist)

    return (
        parent[0] + dx / dist * step,
        parent[1] + dy / dist * step,
    )


def chain_is_connected(nodes, explorer, radio_range):
    """Geometric line-of-sight-free range model; not a radio propagation model."""
    if not nodes:
        return False

    for i in range(1, len(nodes)):
        if distance(nodes[i - 1], nodes[i]) > radio_range:
            return False

    return distance(nodes[-1], explorer) <= radio_range


def find_crossings(times, signal, threshold):
    """Find downward threshold transitions, not every low-signal sample."""
    low = signal <= threshold
    crossing_indices = np.flatnonzero(
        low & np.r_[True, ~low[:-1]]
    )

    # Ignore a recording that starts already below threshold:
    # there is no observed transition within the recording.
    crossing_indices = [
        int(i) for i in crossing_indices
        if i > 0 and signal[i - 1] > threshold
    ]

    return crossing_indices


def simulate_policy(
    df,
    policy,
    radio_range,
    base,
    probability_cutoff,
    cooldown_sec,
    lead_band,
    prediction_window_sec,
    observation_horizon_sec,
):
    df = (
        df.sort_values("elapsed_sec")
        .drop_duplicates("elapsed_sec", keep="first")
        .reset_index(drop=True)
    )

    times = df["elapsed_sec"].to_numpy(dtype=float)
    signal = df["signal_quality"].to_numpy(dtype=float)
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

    crossing_indices = find_crossings(times, signal, threshold)
    crossing_times = times[crossing_indices]

    # Base station is assumed to be at the configured coordinates.
    nodes = [base]
    connected = np.zeros(len(times), dtype=bool)

    dispatches = []
    pending = None
    last_dispatch_time = -np.inf

    for i, t in enumerate(times):
        explorer = (float(xy[i, 0]), float(xy[i, 1]))

        # Complete a pending deployment if its arrival time has passed.
        if pending is not None and pending["arrival_time"] <= t:
            parent_index = pending["parent_index"]
            target = pending["target"]
            parent = nodes[parent_index]

            # Only attach to the current end of the relay chain.
            if (
                parent_index == len(nodes) - 1
                and distance(parent, target) <= radio_range
            ):
                nodes.append(target)
                pending["log"]["deployment_status"] = "deployed"
            else:
                pending["log"]["deployment_status"] = (
                    "placement_link_out_of_range"
                )

            pending = None

        connected[i] = chain_is_connected(
            nodes, explorer, radio_range
        )

        if pending is not None:
            continue

        if t - last_dispatch_time < cooldown_sec:
            continue

        if policy == "reactive":
            trigger = (
                i > 0
                and signal[i - 1] > threshold
                and signal[i] <= threshold
            )

        elif policy == "fixed_lead":
            trigger = (
                signal[i] > threshold
                and signal[i] <= threshold + lead_band
                and changes[i] < 0
            )

        elif policy == "predictive":
            predicted = predicted_times[i]
            trigger = (
                signal[i] > threshold
                and probs[i] >= probability_cutoff
                and np.isfinite(predicted)
                and deployment + margin
                <= predicted
                <= deployment + margin + prediction_window_sec
            )

        else:
            raise ValueError(f"Unknown policy: {policy}")

        if not trigger:
            continue

        target = next_relay_position(
            nodes[-1], explorer, placement_distance
        )

        if target is None:
            continue

        arrival_time = t + deployment + reconnect

        log = {
            "recording_id": df["recording_id"].iloc[0],
            "signal_threshold": threshold,
            "policy": policy,
            "dispatch_time_sec": float(t),
            "arrival_time_sec": float(arrival_time),
            "deployment_time_sec": deployment,
            "reconnect_time_sec": reconnect,
            "predicted_crossing_probability": float(probs[i]),
            "predicted_time_to_crossing_sec": float(predicted_times[i])
            if np.isfinite(predicted_times[i]) else np.nan,
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

    # A relay whose arrival is beyond the recorded trajectory is censored.
    # Any other still-pending relay has no arrival observation either.
    if pending is not None:
        pending["log"]["deployment_status"] = (
            "arrival_not_observed"
        )

    # Classify dispatches against the next observed threshold crossing.
    # This is a signal-crossing diagnostic, not proof of a true outage.
    for d in dispatches:
        t = d["dispatch_time_sec"]
        j = int(np.searchsorted(crossing_times, t, side="left"))

        if j < len(crossing_times):
            lead = crossing_times[j] - t
        else:
            lead = np.inf

        if lead <= observation_horizon_sec:
            if d["arrival_time_sec"] <= crossing_times[j]:
                d["crossing_outcome"] = "arrival_before_crossing"
            else:
                d["crossing_outcome"] = "arrival_after_crossing"
        elif (
            times[-1] - t >= observation_horizon_sec
        ):
            d["crossing_outcome"] = "false_alarm_complete_window"
        else:
            d["crossing_outcome"] = "censored_observation_window"

    # Time-weighted connectivity and outage durations.
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

    # A geometric connection at a signal crossing is not the same as
    # a successful predictive warning; report it separately.
    return {
        "recording_id": df["recording_id"].iloc[0],
        "policy": policy,
        "signal_threshold": threshold,
        "relay_speed_mps": float(df["relay_speed_mps"].iloc[0]),
        "deployment_time_sec": deployment,
        "reconnect_time_sec": reconnect,
        "dispatch_distance_m": placement_distance,
        "safety_margin_sec": margin,
        "radio_range_m": radio_range,
        "probability_cutoff": (
            probability_cutoff if policy == "predictive" else np.nan
        ),
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
            d["deployment_status"] == "deployed" for d in dispatches
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


def main():
    parser = argparse.ArgumentParser(
        description="Simulate predictive relay deployment policies."
    )
    parser.add_argument("--radio-range-m", type=float, default=10.0)
    parser.add_argument("--base-x", type=float, default=0.0)
    parser.add_argument("--base-y", type=float, default=0.0)
    parser.add_argument("--probability-cutoff", type=float, default=0.70)
    parser.add_argument("--cooldown-sec", type=float, default=30.0)
    parser.add_argument("--lead-band", type=float, default=10.0)
    parser.add_argument("--prediction-window-sec", type=float, default=30.0)
    parser.add_argument("--observation-horizon-sec", type=float, default=120.0)
    args = parser.parse_args()

    if args.radio_range_m <= 0:
        parser.error("--radio-range-m must be positive")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    base = (args.base_x, args.base_y)

    details = []
    all_dispatches = []

    grouped = data.groupby(SCENARIO_KEYS, sort=False, dropna=False)
    total_groups = len(grouped)

    for group_number, (_, trajectory) in enumerate(grouped, start=1):
        for policy in POLICIES:
            result = simulate_policy(
                trajectory,
                policy=policy,
                radio_range=args.radio_range_m,
                base=base,
                probability_cutoff=args.probability_cutoff,
                cooldown_sec=args.cooldown_sec,
                lead_band=args.lead_band,
                prediction_window_sec=args.prediction_window_sec,
                observation_horizon_sec=args.observation_horizon_sec,
            )

            dispatch_log = result.pop("dispatch_log")
            details.append(result)
            all_dispatches.extend(dispatch_log)

        if group_number % 25 == 0 or group_number == total_groups:
            print(f"Processed {group_number}/{total_groups} trajectories")

    details_df = pd.DataFrame(details)
    dispatch_df = pd.DataFrame(all_dispatches)

    details_df.to_csv(
        OUTPUT_DIR / "recording_policy_details.csv", index=False
    )
    dispatch_df.to_csv(
        OUTPUT_DIR / "dispatch_log.csv", index=False
    )

    # Pooled results: ratios are calculated from summed numerators
    # and denominators, not from averages of individual percentages.
    summary_rows = []

    for policy, group in details_df.groupby("policy"):
        crossings = int(group["observed_crossing_events"].sum())
        connected_crossings = int(group["connected_at_crossing"].sum())
        total_time = float(group["total_observation_sec"].sum())
        connected_time = float(group["connected_time_sec"].sum())
        dispatches = int(group["total_dispatches"].sum())
        deployed = int(group["relays_deployed"].sum())

        summary_rows.append({
            "policy": policy,
            "radio_range_m": args.radio_range_m,
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
            "total_dispatches": dispatches,
            "relays_deployed": deployed,
            "deployed_per_dispatch_pct": (
                100 * deployed / dispatches if dispatches else np.nan
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

    print("\nSIMULATION SUMMARY")
    print(summary.round(2).to_string(index=False))
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("\nIMPORTANT: connectivity uses an assumed geometric radio range.")
    print("These are simulated results, not measured radio performance.")


if __name__ == "__main__":
    main()
