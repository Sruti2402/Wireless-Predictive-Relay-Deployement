
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT / "data" / "processed" / "script29"
    / "two_stage_oof_predictions.csv"
)
OUTPUT_DIR = ROOT / "data" / "processed" / "script31"

HORIZON_SEC = 120.0
COOLDOWN_SEC = 30.0
FIXED_LEAD_BAND = 10.0
PROBABILITY_CUTOFFS = [0.50, 0.70, 0.90]

SCENARIO_COLUMNS = [
    "signal_threshold",
    "relay_speed_mps",
    "setup_time_sec",
    "reconnect_time_sec",
    "dispatch_distance_m",
    "safety_margin_sec",
]

GROUP_COLUMNS = ["recording_id", *SCENARIO_COLUMNS]


def number(value):
    try:
        result = float(value)
        return result if np.isfinite(result) else np.nan
    except (TypeError, ValueError):
        return np.nan


def get_events(group):
    group = group.sort_values("elapsed_sec").reset_index(drop=True)
    signal = group["signal_quality"].to_numpy(float)
    times = group["elapsed_sec"].to_numpy(float)
    threshold = float(group["signal_threshold"].iloc[0])

    events = []

    for i in range(1, len(group)):
        if not (np.isfinite(signal[i - 1]) and np.isfinite(signal[i])):
            continue

        if signal[i - 1] > threshold and signal[i] <= threshold:
            events.append({
                "event_id": len(events) + 1,
                "event_time_sec": times[i],
            })

    return events


def create_dispatches(group, events, policy, cutoff=None):
    group = group.sort_values("elapsed_sec").reset_index(drop=True)
    threshold = float(group["signal_threshold"].iloc[0])

    if policy == "reactive":
        return [
            {
                "dispatch_time_sec": e["event_time_sec"],
                "trigger_reason": "Signal crossed threshold",
                "predicted_probability": np.nan,
                "predicted_time_sec": np.nan,
                "is_reactive": True,
            }
            for e in events
        ]

    deployment = float(group["deployment_time_sec"].iloc[0])
    margin = float(group["safety_margin_sec"].iloc[0])
    last_dispatch = -np.inf
    dispatches = []

    for row in group.itertuples(index=False):
        now = number(row.elapsed_sec)
        signal = number(row.signal_quality)

        if not np.isfinite(now) or not np.isfinite(signal):
            continue
        if signal <= threshold:
            continue
        if now - last_dispatch < COOLDOWN_SEC:
            continue

        trigger = False
        reason = ""

        if policy == "fixed_lead":
            change = number(row.signal_5s_change)
            trigger = (
                np.isfinite(change)
                and threshold < signal <= threshold + FIXED_LEAD_BAND
                and change < 0
            )
            reason = "Signal near threshold and declining"

        elif policy == "predictive":
            probability = number(row.crossing_probability_120s)
            predicted_time = number(
                row.predicted_time_to_crossing_sec
            )

            trigger = (
                np.isfinite(probability)
                and np.isfinite(predicted_time)
                and probability >= cutoff
                and deployment + margin <= predicted_time
                <= deployment + margin + 30.0
            )
            reason = "Predicted crossing in dispatch window"

        if trigger:
            dispatches.append({
                "dispatch_time_sec": now,
                "trigger_reason": reason,
                "predicted_probability": number(
                    row.crossing_probability_120s
                ),
                "predicted_time_sec": number(
                    row.predicted_time_to_crossing_sec
                ),
            })
            last_dispatch = now

    return dispatches


def match_events_and_dispatches(
    group, events, dispatches
):
    """
    One-to-one matching.

    A dispatch can cover an event only if it occurs early enough
    for deployment plus safety margin, and no more than the
    evaluation horizon before the event.
    """
    deployment = float(group["deployment_time_sec"].iloc[0])
    margin = float(group["safety_margin_sec"].iloc[0])
    recording_end = float(group["elapsed_sec"].max())

    dispatches = sorted(
        [dict(d, dispatch_id=i + 1)
         for i, d in enumerate(dispatches)],
        key=lambda d: d["dispatch_time_sec"],
    )

    used_dispatch_ids = set()
    event_results = []
    dispatch_results = []

    # Process events in chronological order. Choose the latest
    # still-available dispatch that can feasibly cover each event.
    for event in events:
        event_time = event["event_time_sec"]

        candidates = [
            d for d in dispatches
            if d["dispatch_id"] not in used_dispatch_ids
            and not d.get("is_reactive", False)
            and 0 <= event_time - d["dispatch_time_sec"] <= HORIZON_SEC
            and d["dispatch_time_sec"] + deployment + margin
            <= event_time
        ]

        selected = max(
            candidates,
            key=lambda d: d["dispatch_time_sec"],
        ) if candidates else None

        if selected:
            used_dispatch_ids.add(selected["dispatch_id"])
            lead = event_time - selected["dispatch_time_sec"]
            event_results.append({
                **event,
                "event_covered": True,
                "matched_dispatch_id": selected["dispatch_id"],
                "warning_lead_time_sec": lead,
            })
        else:
            event_results.append({
                **event,
                "event_covered": False,
                "matched_dispatch_id": np.nan,
                "warning_lead_time_sec": np.nan,
            })

    # Classify every dispatch not used for coverage.
    for dispatch in dispatches:
        t = dispatch["dispatch_time_sec"]
        dispatch_id = dispatch["dispatch_id"]

        future_events = [
            e for e in events
            if 0 <= e["event_time_sec"] - t <= HORIZON_SEC
        ]

        feasible_future_events = [
            e for e in future_events
            if t + deployment + margin <= e["event_time_sec"]
        ]

        if dispatch_id in used_dispatch_ids:
            status = "matched_feasible"
            next_event_time = min(
                (
                    e["event_time_sec"] for e in events
                    if e["event_time_sec"] >= t
                ),
                default=np.nan,
            )
        elif feasible_future_events:
            # A feasible event exists, but another dispatch
            # already claimed the event or an earlier one.
            status = "duplicate_or_excess_dispatch"
            next_event_time = feasible_future_events[0][
                "event_time_sec"
            ]
        elif future_events:
            status = "too_late_for_next_crossing"
            next_event_time = future_events[0]["event_time_sec"]
        elif recording_end - t >= HORIZON_SEC:
            status = "false_alarm_complete_window"
            next_event_time = np.nan
        else:
            status = "censored_observation_window"
            next_event_time = np.nan

        dispatch_results.append({
            **dispatch,
            "next_event_time_sec": next_event_time,
            "dispatch_status": status,
            "deployment_time_sec": deployment,
            "safety_margin_sec": margin,
        })

    return event_results, dispatch_results


def evaluate_trajectory(group, scenario):
    group = group.sort_values("elapsed_sec").reset_index(drop=True)
    events = get_events(group)

    recording_id = group["recording_id"].iloc[0]
    deployment = float(group["deployment_time_sec"].iloc[0])
    margin = float(group["safety_margin_sec"].iloc[0])

    policies = [
        ("reactive", None),
        ("fixed_lead", None),
        *[("predictive", c) for c in PROBABILITY_CUTOFFS],
    ]

    summaries = []
    event_rows = []
    dispatch_rows = []

    for policy, cutoff in policies:
        label = (
            f"predictive_p{cutoff:.2f}"
            if policy == "predictive"
            else policy
        )

        dispatches = create_dispatches(
            group, events, policy, cutoff
        )

        matched_events, matched_dispatches = (
            match_events_and_dispatches(
                group, events, dispatches
            )
        )

        for row in matched_events:
            event_rows.append({
                **scenario,
                "recording_id": recording_id,
                "policy": label,
                **row,
            })

        for row in matched_dispatches:
            dispatch_rows.append({
                **scenario,
                "recording_id": recording_id,
                "policy": label,
                **row,
            })

        n_events = len(matched_events)
        covered = sum(
            bool(e["event_covered"]) for e in matched_events
        )

        status_counts = pd.Series([
            d["dispatch_status"] for d in matched_dispatches
        ]).value_counts().to_dict()

        n_dispatches = len(matched_dispatches)
        feasible = status_counts.get("matched_feasible", 0)

        leads = [
            e["warning_lead_time_sec"]
            for e in matched_events
            if e["event_covered"]
            and np.isfinite(e["warning_lead_time_sec"])
        ]

        summaries.append({
            **scenario,
            "recording_id": recording_id,
            "policy": label,
            "deployment_time_sec": deployment,
            "safety_margin_sec": margin,
            "observed_crossing_events": n_events,
            "covered_events": covered,
            "missed_events": n_events - covered,
            "event_coverage_percent": (
                100 * covered / n_events if n_events else np.nan
            ),
            "total_dispatches": n_dispatches,
            "matched_feasible_dispatches": feasible,
            "dispatch_feasible_percent": (
                100 * feasible / n_dispatches
                if n_dispatches else np.nan
            ),
            "duplicate_or_excess_dispatches": status_counts.get(
                "duplicate_or_excess_dispatch", 0
            ),
            "too_late_dispatches": status_counts.get(
                "too_late_for_next_crossing", 0
            ),
            "false_alarms_complete_window": status_counts.get(
                "false_alarm_complete_window", 0
            ),
            "censored_dispatches": status_counts.get(
                "censored_observation_window", 0
            ),
            "mean_warning_lead_time_sec": (
                float(np.mean(leads)) if leads else np.nan
            ),
        })

    return summaries, event_rows, dispatch_rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Cannot find {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE, low_memory=False)

    required = [
        *GROUP_COLUMNS,
        "elapsed_sec",
        "signal_quality",
        "signal_5s_change",
        "crossing_probability_120s",
        "predicted_time_to_crossing_sec",
        "deployment_time_sec",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    numeric = [
        c for c in required
        if c != "recording_id"
    ]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=[
            *GROUP_COLUMNS,
            "elapsed_sec",
            "signal_quality",
            "deployment_time_sec",
        ]
    ).copy()

    df["crossing_probability_120s"] = (
        df["crossing_probability_120s"].clip(0, 1)
    )

    summaries, event_rows, dispatch_rows = [], [], []

    grouped = df.groupby(
        GROUP_COLUMNS, sort=False, dropna=False
    )
    print(f"Input rows: {len(df):,}")
    print(f"Trajectories: {grouped.ngroups:,}")

    for i, (key, group) in enumerate(grouped, start=1):
        if not isinstance(key, tuple):
            key = (key,)

        scenario = dict(zip(GROUP_COLUMNS, key))
        s, e, d = evaluate_trajectory(group, scenario)

        summaries.extend(s)
        event_rows.extend(e)
        dispatch_rows.extend(d)

        if i % 50 == 0 or i == grouped.ngroups:
            print(f"Processed {i}/{grouped.ngroups}")

    summary = pd.DataFrame(summaries)
    events = pd.DataFrame(event_rows)
    dispatches = pd.DataFrame(dispatch_rows)

    summary.to_csv(
        OUTPUT_DIR / "recording_policy_details.csv", index=False
    )
    events.to_csv(
        OUTPUT_DIR / "event_details.csv", index=False
    )
    dispatches.to_csv(
        OUTPUT_DIR / "dispatch_log.csv", index=False
    )

    # Aggregate counts across trajectories, then calculate ratios
    # from pooled counts instead of averaging percentages.
    overall = (
        summary.groupby(
            [*SCENARIO_COLUMNS, "policy"],
            dropna=False,
            as_index=False,
        )
        .agg(
            observed_crossing_events=("observed_crossing_events", "sum"),
            covered_events=("covered_events", "sum"),
            missed_events=("missed_events", "sum"),
            total_dispatches=("total_dispatches", "sum"),
            matched_feasible_dispatches=(
                "matched_feasible_dispatches", "sum"
            ),
            duplicate_or_excess_dispatches=(
                "duplicate_or_excess_dispatches", "sum"
            ),
            too_late_dispatches=("too_late_dispatches", "sum"),
            false_alarms_complete_window=(
                "false_alarms_complete_window", "sum"
            ),
            censored_dispatches=("censored_dispatches", "sum"),
            mean_warning_lead_time_sec=(
                "mean_warning_lead_time_sec", "mean"
            ),
        )
    )

    overall["event_coverage_percent"] = np.where(
        overall["observed_crossing_events"] > 0,
        100 * overall["covered_events"]
        / overall["observed_crossing_events"],
        np.nan,
    )

    overall["dispatch_feasible_percent"] = np.where(
        overall["total_dispatches"] > 0,
        100 * overall["matched_feasible_dispatches"]
        / overall["total_dispatches"],
        np.nan,
    )

    overall.to_csv(OUTPUT_DIR / "policy_summary.csv", index=False)

    print("\nMEAN METRICS BY POLICY")
    metrics = (
        summary.groupby("policy")
        .agg(
            event_coverage_percent=("event_coverage_percent", "mean"),
            dispatch_feasible_percent=("dispatch_feasible_percent", "mean"),
            dispatches=("total_dispatches", "mean"),
            duplicates=("duplicate_or_excess_dispatches", "mean"),
            late_dispatches=("too_late_dispatches", "mean"),
            false_alarms=("false_alarms_complete_window", "mean"),
            censored=("censored_dispatches", "mean"),
            warning_lead_sec=("mean_warning_lead_time_sec", "mean"),
        )
        .round(2)
    )
    print(metrics.to_string())

    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("Reminder: these are simulated decisions on observed signal crossings.")


if __name__ == "__main__":
    main()
