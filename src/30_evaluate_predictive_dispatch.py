
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# SCRIPT 30: EVENT-BASED PREDICTIVE RELAY DISPATCH EVALUATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "script29"
    / "two_stage_oof_predictions.csv"
)

OUTPUT_DIR = ROOT / "data" / "processed" / "script30"

HORIZON_SEC = 120.0
DISPATCH_COOLDOWN_SEC = 30.0

# Classifier probability cutoffs to evaluate.
PROBABILITY_CUTOFFS = [0.50, 0.70, 0.90]

# Simple comparison controller:
# dispatch when signal is approaching the threshold and declining.
FIXED_LEAD_BAND = 10.0

SCENARIO_COLUMNS = [
    "signal_threshold",
    "relay_speed_mps",
    "setup_time_sec",
    "reconnect_time_sec",
    "dispatch_distance_m",
    "safety_margin_sec",
]

REQUIRED_COLUMNS = [
    "recording_id",
    "elapsed_sec",
    "signal_quality",
    "signal_5s_change",
    "crossing_probability_120s",
    "predicted_time_to_crossing_sec",
    "deployment_time_sec",
    *SCENARIO_COLUMNS,
]


def safe_mean(values):
    values = pd.Series(values).dropna()
    return float(values.mean()) if len(values) else np.nan


def find_crossing_events(group):
    """
    Identify distinct observed downward signal-threshold crossings.

    A crossing occurs when the signal moves from above the threshold
    to at or below it. A recording that starts below the threshold
    does not automatically count as a new crossing.
    """
    group = group.sort_values("elapsed_sec").reset_index(drop=True)

    threshold = float(group["signal_threshold"].iloc[0])
    signal = group["signal_quality"].to_numpy(dtype=float)
    times = group["elapsed_sec"].to_numpy(dtype=float)

    valid = np.isfinite(signal) & np.isfinite(times)

    events = []

    for i in range(1, len(group)):
        if not (valid[i - 1] and valid[i]):
            continue

        if signal[i - 1] > threshold and signal[i] <= threshold:
            events.append({
                "event_id": len(events) + 1,
                "event_time_sec": float(times[i]),
                "signal_before_crossing": float(signal[i - 1]),
                "signal_at_crossing": float(signal[i]),
                "signal_threshold": threshold,
            })

    return events


def get_next_event(events, dispatch_time):
    """Return the next crossing within the evaluation horizon."""
    if not events:
        return None

    event_times = np.array(
        [event["event_time_sec"] for event in events],
        dtype=float,
    )

    idx = np.searchsorted(event_times, dispatch_time, side="left")

    if idx >= len(events):
        return None

    event = events[idx]

    if event["event_time_sec"] - dispatch_time > HORIZON_SEC:
        return None

    return event


def generate_dispatches(group, events, policy, probability_cutoff=None):
    """
    Generate dispatch decisions for one recording/scenario.

    Policies:
      1. Reactive: dispatch at the observed crossing.
      2. Fixed lead: signal is within a fixed band above threshold
         and its recent change is negative.
      3. Predictive: probability and predicted crossing time both
         satisfy the controller's dispatch conditions.
    """
    group = group.sort_values("elapsed_sec").reset_index(drop=True)

    threshold = float(group["signal_threshold"].iloc[0])
    deployment_time = float(group["deployment_time_sec"].iloc[0])
    safety_margin = float(group["safety_margin_sec"].iloc[0])

    dispatches = []

    # The reactive policy fires once at each observed crossing.
    if policy == "reactive":
        for event in events:
            dispatches.append({
                "policy": policy,
                "dispatch_time_sec": event["event_time_sec"],
                "probability_cutoff": np.nan,
                "predicted_probability": np.nan,
                "predicted_time_sec": np.nan,
                "trigger_reason": "Observed signal crossing",
            })

        return dispatches

    last_dispatch_time = -np.inf

    for row in group.itertuples(index=False):
        now = float(row.elapsed_sec)
        signal = float(row.signal_quality)

        if not np.isfinite(now) or not np.isfinite(signal):
            continue

        # Do not dispatch after the signal has already crossed.
        if signal <= threshold:
            continue

        # Avoid repeated dispatches from adjacent samples.
        if now - last_dispatch_time < DISPATCH_COOLDOWN_SEC:
            continue

        should_dispatch = False
        reason = ""

        if policy == "fixed_lead":
            recent_change = float(row.signal_5s_change)

            if (
                np.isfinite(recent_change)
                and threshold < signal <= threshold + FIXED_LEAD_BAND
                and recent_change < 0
            ):
                should_dispatch = True
                reason = "Signal near threshold and declining"

        elif policy == "predictive":
            probability = float(row.crossing_probability_120s)
            predicted_time = float(row.predicted_time_to_crossing_sec)

            if (
                np.isfinite(probability)
                and np.isfinite(predicted_time)
                and probability >= probability_cutoff
                and deployment_time + safety_margin <= predicted_time
                and predicted_time <= deployment_time + safety_margin + 30.0
            ):
                should_dispatch = True
                reason = "Predicted crossing near deployment deadline"

        if should_dispatch:
            probability = (
                float(row.crossing_probability_120s)
                if np.isfinite(row.crossing_probability_120s)
                else np.nan
            )

            predicted_time = (
                float(row.predicted_time_to_crossing_sec)
                if np.isfinite(row.predicted_time_to_crossing_sec)
                else np.nan
            )

            dispatches.append({
                "policy": policy,
                "dispatch_time_sec": now,
                "probability_cutoff": (
                    probability_cutoff
                    if policy == "predictive"
                    else np.nan
                ),
                "predicted_probability": probability,
                "predicted_time_sec": predicted_time,
                "trigger_reason": reason,
            })

            last_dispatch_time = now

    return dispatches


def evaluate_scenario(group, scenario_key):
    """
    Evaluate all policies for a single recording and configuration.
    """
    group = group.sort_values("elapsed_sec").reset_index(drop=True)

    recording_id = group["recording_id"].iloc[0]
    threshold = float(group["signal_threshold"].iloc[0])
    deployment_time = float(group["deployment_time_sec"].iloc[0])
    safety_margin = float(group["safety_margin_sec"].iloc[0])

    events = find_crossing_events(group)

    policies = [
        ("reactive", None),
        ("fixed_lead", None),
    ]

    policies.extend(
        ("predictive", cutoff)
        for cutoff in PROBABILITY_CUTOFFS
    )

    all_event_results = []
    all_dispatch_results = []
    all_summary_results = []

    for policy, cutoff in policies:
        dispatches = generate_dispatches(
            group=group,
            events=events,
            policy=policy,
            probability_cutoff=cutoff,
        )

        matched_dispatches = {}

        for dispatch_idx, dispatch in enumerate(dispatches):
            dispatch_time = dispatch["dispatch_time_sec"]

            next_event = get_next_event(events, dispatch_time)

            if next_event is None:
                event_id = np.nan
                event_time = np.nan
                lead_time = np.nan
                feasible = False
                matched = False
            else:
                event_id = next_event["event_id"]
                event_time = next_event["event_time_sec"]
                lead_time = event_time - dispatch_time

                # A relay is feasible only if deployment, reconnection,
                # and the requested safety margin fit before crossing.
                feasible = (
                    dispatch_time
                    + deployment_time
                    + safety_margin
                    <= event_time
                )
                matched = True

                matched_dispatches.setdefault(
                    next_event["event_id"], []
                ).append({
                    "dispatch_idx": dispatch_idx,
                    "dispatch_time_sec": dispatch_time,
                    "feasible": feasible,
                    "lead_time_sec": lead_time,
                })

            all_dispatch_results.append({
                **scenario_key,
                "recording_id": recording_id,
                "policy": (
                    f"predictive_p{cutoff:.2f}"
                    if policy == "predictive"
                    else policy
                ),
                "dispatch_time_sec": dispatch_time,
                "deployment_time_sec": deployment_time,
                "safety_margin_sec": safety_margin,
                "predicted_probability": dispatch[
                    "predicted_probability"
                ],
                "probability_cutoff": dispatch["probability_cutoff"],
                "predicted_time_to_crossing_sec": dispatch[
                    "predicted_time_sec"
                ],
                "next_event_id": event_id,
                "next_event_time_sec": event_time,
                "warning_lead_time_sec": lead_time,
                "predicted_arrival_time_sec": (
                    dispatch_time + deployment_time
                ),
                "feasible_for_next_event": bool(feasible),
                "matched_to_event": bool(matched),
                "trigger_reason": dispatch["trigger_reason"],
            })

        covered_events = 0
        event_lead_times = []
        missed_events = 0

        for event in events:
            candidates = matched_dispatches.get(
                event["event_id"], []
            )

            feasible_candidates = [
                item for item in candidates if item["feasible"]
            ]

            covered = len(feasible_candidates) > 0

            if covered:
                covered_events += 1

                # Use the latest feasible dispatch before this event:
                # this reflects the last opportunity to deploy in time.
                selected = max(
                    feasible_candidates,
                    key=lambda item: item["dispatch_time_sec"],
                )
                event_lead_times.append(selected["lead_time_sec"])
            else:
                missed_events += 1

            all_event_results.append({
                **scenario_key,
                "recording_id": recording_id,
                "policy": (
                    f"predictive_p{cutoff:.2f}"
                    if policy == "predictive"
                    else policy
                ),
                "event_id": event["event_id"],
                "event_time_sec": event["event_time_sec"],
                "signal_threshold": threshold,
                "deployment_time_sec": deployment_time,
                "safety_margin_sec": safety_margin,
                "dispatches_matched_to_event": len(candidates),
                "feasible_dispatches_for_event": len(
                    feasible_candidates
                ),
                "event_covered": covered,
                "event_missed": not covered,
                "warning_lead_time_sec": (
                    max(event_lead_times[-1:], default=np.nan)
                    if covered
                    else np.nan
                ),
            })

        total_dispatches = len(dispatches)
        feasible_dispatches = sum(
            1
            for item in all_dispatch_results[-total_dispatches:]
            if item["recording_id"] == recording_id
            and item["policy"] == (
                f"predictive_p{cutoff:.2f}"
                if policy == "predictive"
                else policy
            )
            and item["feasible_for_next_event"]
        ) if total_dispatches else 0

        unmatched_dispatches = sum(
            1
            for item in all_dispatch_results[-total_dispatches:]
            if item["recording_id"] == recording_id
            and item["policy"] == (
                f"predictive_p{cutoff:.2f}"
                if policy == "predictive"
                else policy
            )
            and not item["matched_to_event"]
        ) if total_dispatches else 0

        duplicate_dispatches = sum(
            max(0, len(items) - 1)
            for items in matched_dispatches.values()
        )

        all_summary_results.append({
            **scenario_key,
            "recording_id": recording_id,
            "policy": (
                f"predictive_p{cutoff:.2f}"
                if policy == "predictive"
                else policy
            ),
            "signal_threshold": threshold,
            "relay_speed_mps": float(
                group["relay_speed_mps"].iloc[0]
            ),
            "setup_time_sec": float(
                group["setup_time_sec"].iloc[0]
            ),
            "reconnect_time_sec": float(
                group["reconnect_time_sec"].iloc[0]
            ),
            "dispatch_distance_m": float(
                group["dispatch_distance_m"].iloc[0]
            ),
            "safety_margin_sec": safety_margin,
            "deployment_time_sec": deployment_time,
            "observed_crossing_events": len(events),
            "total_dispatches": total_dispatches,
            "feasible_dispatches": feasible_dispatches,
            "dispatch_feasible_percent": (
                100 * feasible_dispatches / total_dispatches
                if total_dispatches else np.nan
            ),
            "covered_events": covered_events,
            "missed_events": missed_events,
            "event_coverage_percent": (
                100 * covered_events / len(events)
                if events else np.nan
            ),
            "unmatched_dispatches": unmatched_dispatches,
            "duplicate_dispatches": duplicate_dispatches,
            "mean_warning_lead_time_sec": safe_mean(
                event_lead_times
            ),
        })

    return all_summary_results, all_event_results, all_dispatch_results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input predictions not found:\n{INPUT_FILE}"
        )

    print("=" * 76)
    print("SCRIPT 30: PREDICTIVE RELAY DISPATCH EVALUATION")
    print("=" * 76)

    df = pd.read_csv(INPUT_FILE, low_memory=False)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing:
        raise ValueError(
            "The Script 29 prediction file is missing required columns:\n"
            + "\n".join(missing)
            + "\n\nAvailable columns:\n"
            + ", ".join(df.columns)
        )

    # Retain complete scenario metadata. Rows without essential signal
    # or prediction fields cannot trigger a model-based dispatch.
    numeric_columns = [
        "elapsed_sec",
        "signal_quality",
        "signal_5s_change",
        "crossing_probability_120s",
        "predicted_time_to_crossing_sec",
        "deployment_time_sec",
        *SCENARIO_COLUMNS,
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=[
            "recording_id",
            "elapsed_sec",
            "signal_quality",
            "deployment_time_sec",
            *SCENARIO_COLUMNS,
        ]
    ).copy()

    # Keep the probabilities within the mathematically valid range.
    df["crossing_probability_120s"] = (
        df["crossing_probability_120s"].clip(0, 1)
    )

    # One independent trajectory per recording/configuration.
    group_columns = ["recording_id", *SCENARIO_COLUMNS]

    summary_rows = []
    event_rows = []
    dispatch_rows = []

    grouped = df.groupby(group_columns, sort=False, dropna=False)
    total_groups = grouped.ngroups

    print(f"Input rows: {len(df):,}")
    print(f"Recording/configuration trajectories: {total_groups:,}")
    print(f"Probability cutoffs: {PROBABILITY_CUTOFFS}")
    print(f"Dispatch cooldown: {DISPATCH_COOLDOWN_SEC:.0f}s")
    print()

    for idx, (key, group) in enumerate(grouped, start=1):
        if not isinstance(key, tuple):
            key = (key,)

        scenario_key = dict(zip(group_columns, key))

        summaries, events, dispatches = evaluate_scenario(
            group=group,
            scenario_key=scenario_key,
        )

        summary_rows.extend(summaries)
        event_rows.extend(events)
        dispatch_rows.extend(dispatches)

        if idx % 50 == 0 or idx == total_groups:
            print(f"Processed {idx:,}/{total_groups:,} trajectories")

    summary_df = pd.DataFrame(summary_rows)
    event_df = pd.DataFrame(event_rows)
    dispatch_df = pd.DataFrame(dispatch_rows)

    # Aggregate event and dispatch counts across held-out recordings.
    aggregation = {
        "observed_crossing_events": "sum",
        "total_dispatches": "sum",
        "feasible_dispatches": "sum",
        "covered_events": "sum",
        "missed_events": "sum",
        "unmatched_dispatches": "sum",
        "duplicate_dispatches": "sum",
    }

    overall = (
        summary_df.groupby(
            SCENARIO_COLUMNS + ["policy"],
            dropna=False,
            as_index=False,
        )
        .agg(aggregation)
    )

    overall["event_coverage_percent"] = np.where(
        overall["observed_crossing_events"] > 0,
        100 * overall["covered_events"]
        / overall["observed_crossing_events"],
        np.nan,
    )

    overall["dispatch_feasible_percent"] = np.where(
        overall["total_dispatches"] > 0,
        100 * overall["feasible_dispatches"]
        / overall["total_dispatches"],
        np.nan,
    )

    # Means are reported across recording/configuration trajectories;
    # these are descriptive, not independent-event confidence intervals.
    means = (
        summary_df.groupby(
            SCENARIO_COLUMNS + ["policy"],
            dropna=False,
            as_index=False,
        )
        .agg(
            mean_warning_lead_time_sec=(
                "mean_warning_lead_time_sec", "mean"
            ),
            mean_trajectory_coverage_percent=(
                "event_coverage_percent", "mean"
            ),
            mean_trajectory_dispatch_feasible_percent=(
                "dispatch_feasible_percent", "mean"
            ),
        )
    )

    overall = overall.merge(
        means,
        on=SCENARIO_COLUMNS + ["policy"],
        how="left",
    )

    summary_path = OUTPUT_DIR / "policy_summary.csv"
    event_path = OUTPUT_DIR / "event_details.csv"
    dispatch_path = OUTPUT_DIR / "dispatch_log.csv"
    trajectory_path = OUTPUT_DIR / "recording_policy_details.csv"

    overall.to_csv(summary_path, index=False)
    event_df.to_csv(event_path, index=False)
    dispatch_df.to_csv(dispatch_path, index=False)
    summary_df.to_csv(trajectory_path, index=False)

    print("\n" + "=" * 76)
    print("TOP CONFIGURATIONS BY EVENT COVERAGE")
    print("=" * 76)

    display_columns = [
        "policy",
        "signal_threshold",
        "relay_speed_mps",
        "setup_time_sec",
        "reconnect_time_sec",
        "safety_margin_sec",
        "observed_crossing_events",
        "event_coverage_percent",
        "dispatch_feasible_percent",
        "total_dispatches",
        "unmatched_dispatches",
        "duplicate_dispatches",
        "mean_warning_lead_time_sec",
    ]

    ranked = overall.sort_values(
        ["event_coverage_percent", "dispatch_feasible_percent"],
        ascending=[False, False],
        na_position="last",
    )

    print(
        ranked[display_columns]
        .head(20)
        .to_string(index=False)
    )

    print("\nSaved outputs:")
    print(f"  Policy summary: {summary_path}")
    print(f"  Event details:  {event_path}")
    print(f"  Dispatch log:   {dispatch_path}")
    print(f"  Per trajectory: {trajectory_path}")

    print("\nInterpretation reminders:")
    print("- Events are observed signal-threshold crossings, not confirmed outages.")
    print("- Feasibility is calculated from assumed deployment time and safety margin.")
    print("- Coverage does not prove that a relay restored or maintained the link.")
    print("- Recording-level and scenario-level dependence limits statistical claims.")


if __name__ == "__main__":
    main()
