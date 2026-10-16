
from pathlib import Path
import itertools
import warnings

import numpy as np
import pandas as pd


# ============================================================
# PATHS AND SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

PREDICTION_FILE = PROCESSED / "trend_model_oof_predictions.csv"
CROSSING_FILE = PROCESSED / "observed_threshold_crossings.csv"

OUTPUT_DIR = PROCESSED / "script22"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROBABILITY_CUTOFFS = [0.30, 0.40, 0.50, 0.60, 0.70]

PREDICTION_HORIZON_SEC = 30.0
ALERT_COOLDOWN_SEC = 30.0

RELAY_DISTANCES_M = [5, 10, 15]
RELAY_SPEEDS_MPS = [0.1, 0.2, 0.4]
SETUP_TIMES_SEC = [2, 5]
RECONNECTION_TIMES_SEC = [1, 3]
SAFETY_MARGINS_SEC = [0, 2]


# ============================================================
# LOAD DATA
# ============================================================

def load_csv(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}\n"
            "Check that Script 21 generated this file."
        )

    return pd.read_csv(path)


def find_column(df, candidates, description):
    for name in candidates:
        if name in df.columns:
            return name

    raise ValueError(
        f"Could not identify {description}.\n"
        f"Expected one of: {candidates}\n"
        f"Available columns: {df.columns.tolist()}"
    )


def load_inputs():
    predictions = load_csv(PREDICTION_FILE)
    crossings = load_csv(CROSSING_FILE)

    required = {
        "threshold",
        "horizon_sec",
        "model",
        "feature_set",
        "recording_id",
        "elapsed_sec",
        "risk_probability",
    }

    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(
            f"Prediction file is missing columns: {sorted(missing)}"
        )

    crossing_time_col = find_column(
        crossings,
        [
            "crossing_time_sec",
            "elapsed_sec",
            "crossing_sec",
            "time_sec",
            "event_time_sec",
        ],
        "crossing timestamp",
    )

    crossing_recording_col = find_column(
        crossings,
        ["recording_id"],
        "crossing recording ID",
    )

    crossing_threshold_col = find_column(
        crossings,
        ["threshold", "signal_threshold"],
        "crossing threshold",
    )

    predictions = predictions.copy()
    predictions["elapsed_sec"] = pd.to_numeric(
        predictions["elapsed_sec"], errors="coerce"
    )
    predictions["risk_probability"] = pd.to_numeric(
        predictions["risk_probability"], errors="coerce"
    )
    predictions["threshold"] = pd.to_numeric(
        predictions["threshold"], errors="coerce"
    )
    predictions["horizon_sec"] = pd.to_numeric(
        predictions["horizon_sec"], errors="coerce"
    )

    predictions = predictions.dropna(
        subset=[
            "recording_id",
            "elapsed_sec",
            "risk_probability",
            "threshold",
            "horizon_sec",
        ]
    )

    crossings = crossings.copy()
    crossings["event_time_sec"] = pd.to_numeric(
        crossings[crossing_time_col], errors="coerce"
    )
    crossings["threshold"] = pd.to_numeric(
        crossings[crossing_threshold_col], errors="coerce"
    )
    crossings["recording_id"] = (
        crossings[crossing_recording_col].astype(str)
    )

    crossings = crossings.dropna(
        subset=["recording_id", "event_time_sec", "threshold"]
    )

    # If the crossings file has an event type, keep only downward
    # crossings when that information is available.
    event_col = next(
        (
            c for c in ["crossing_type", "direction", "event_type"]
            if c in crossings.columns
        ),
        None,
    )

    if event_col:
        text = crossings[event_col].astype(str).str.lower()
        downward = text.str.contains(
            "down|fall|degrad|cross_below", regex=True
        )
        if downward.any():
            crossings = crossings[downward].copy()

    return predictions, crossings


# ============================================================
# BUILD ALERT EPISODES FOR EACH PROBABILITY CUTOFF
# ============================================================

def build_alerts(group, probability_cutoff):
    group = group.sort_values("elapsed_sec")

    triggered = group[
        group["risk_probability"] >= probability_cutoff
    ]

    if triggered.empty:
        return []

    # Group closely spaced high-risk predictions into one episode.
    # The first prediction in each episode becomes its alert time.
    alerts = []
    last_alert_time = -np.inf

    for row in triggered.itertuples(index=False):
        time_sec = float(row.elapsed_sec)

        if time_sec - last_alert_time >= ALERT_COOLDOWN_SEC:
            alerts.append({
                "alert_time_sec": time_sec,
                "risk_probability": float(row.risk_probability),
            })
            last_alert_time = time_sec

    return alerts


# ============================================================
# MATCH ALERTS TO FUTURE CROSSINGS
# ============================================================

def match_events(alerts, event_times, horizon_sec):
    """
    Each crossing and alert can be used at most once.

    Match the latest eligible preceding alert to each crossing.
    Only crossings within the prediction horizon are eligible.
    """
    alerts = sorted(alerts, key=lambda x: x["alert_time_sec"])
    events = sorted(float(t) for t in event_times)

    used_alerts = set()
    matches = []

    for event_time in events:
        eligible = [
            (i, alert)
            for i, alert in enumerate(alerts)
            if i not in used_alerts
            and 0 < event_time - alert["alert_time_sec"]
            <= horizon_sec
        ]

        if eligible:
            alert_index, alert = max(
                eligible,
                key=lambda pair: pair[1]["alert_time_sec"],
            )
            used_alerts.add(alert_index)

            matches.append({
                "crossing_time_sec": event_time,
                "alert_time_sec": alert["alert_time_sec"],
                "risk_probability": alert["risk_probability"],
                "warning_lead_sec": (
                    event_time - alert["alert_time_sec"]
                ),
                "alert_outcome": "matched",
            })
        else:
            matches.append({
                "crossing_time_sec": event_time,
                "alert_time_sec": np.nan,
                "risk_probability": np.nan,
                "warning_lead_sec": np.nan,
                "alert_outcome": "missed_crossing",
            })

    for i, alert in enumerate(alerts):
        if i not in used_alerts:
            matches.append({
                "crossing_time_sec": np.nan,
                "alert_time_sec": alert["alert_time_sec"],
                "risk_probability": alert["risk_probability"],
                "warning_lead_sec": np.nan,
                "alert_outcome": "unmatched_alert",
            })

    return matches


# ============================================================
# EVALUATE POLICY
# ============================================================

def safe_div(numerator, denominator):
    return numerator / denominator if denominator else np.nan


def main():
    predictions, crossings = load_inputs()

    print("=" * 72)
    print("SCRIPT 22 — DISPATCH POLICY OPTIMIZATION")
    print("=" * 72)
    print(f"Prediction rows: {len(predictions):,}")
    print(f"Observed crossing rows: {len(crossings):,}")
    print(f"Probability cutoffs: {PROBABILITY_CUTOFFS}")

    # Only compare the same horizon and the available signal thresholds.
    predictions = predictions[
        predictions["horizon_sec"] == PREDICTION_HORIZON_SEC
    ].copy()

    prediction_keys = [
        "threshold", "model", "feature_set", "recording_id"
    ]

    crossing_groups = {
        (float(threshold), str(recording_id)): group[
            "event_time_sec"
        ].tolist()
        for (threshold, recording_id), group in crossings.groupby(
            ["threshold", "recording_id"]
        )
    }

    all_matches = []
    summaries = []

    for keys, pred_group in predictions.groupby(
        prediction_keys, dropna=False
    ):
        signal_threshold, model, feature_set, recording_id = keys

        # Keep the recording ID representation consistent.
        recording_id = str(recording_id)

        event_times = crossing_groups.get(
            (float(signal_threshold), recording_id), []
        )

        if not event_times:
            # Do not silently claim success when a group has no events.
            # It can still contribute alert-count information.
            pass

        for cutoff in PROBABILITY_CUTOFFS:
            alerts = build_alerts(pred_group, cutoff)

            matches = match_events(
                alerts,
                event_times,
                PREDICTION_HORIZON_SEC,
            )

            matched = [
                m for m in matches
                if m["alert_outcome"] == "matched"
            ]

            missed = [
                m for m in matches
                if m["alert_outcome"] == "missed_crossing"
            ]

            unmatched = [
                m for m in matches
                if m["alert_outcome"] == "unmatched_alert"
            ]

            # False-alert rate is reported as a fraction of alert
            # episodes, not as a rate per hour.
            total_alerts = len(alerts)
            event_count = len(event_times)

            group_record = {
                "threshold": signal_threshold,
                "model": model,
                "feature_set": feature_set,
                "recording_id": recording_id,
                "probability_cutoff": cutoff,
                "crossings": event_count,
                "alerts": total_alerts,
                "matched_crossings": len(matched),
                "missed_crossings": len(missed),
                "unmatched_alerts": len(unmatched),
                "crossing_coverage": safe_div(
                    len(matched), event_count
                ),
                "alert_precision": safe_div(
                    len(matched), total_alerts
                ),
                "unmatched_alert_fraction": safe_div(
                    len(unmatched), total_alerts
                ),
                "median_warning_sec": (
                    float(np.median([
                        m["warning_lead_sec"] for m in matched
                    ]))
                    if matched else np.nan
                ),
            }
            summaries.append(group_record)

            for match in matches:
                all_matches.append({
                    **{
                        "threshold": signal_threshold,
                        "model": model,
                        "feature_set": feature_set,
                        "recording_id": recording_id,
                        "probability_cutoff": cutoff,
                    },
                    **match,
                })

    summary_df = pd.DataFrame(summaries)
    matches_df = pd.DataFrame(all_matches)

    if summary_df.empty:
        raise ValueError(
            "No results were generated. Check the prediction horizon "
            "and input data."
        )

    summary_df.to_csv(
        OUTPUT_DIR / "policy_by_recording.csv", index=False
    )
    matches_df.to_csv(
        OUTPUT_DIR / "policy_event_matches.csv", index=False
    )

    # Pool counts across recordings rather than averaging per-recording
    # precision/recall values.
    pooled_rows = []

    for keys, group in summary_df.groupby(
        ["threshold", "model", "feature_set", "probability_cutoff"]
    ):
        signal_threshold, model, feature_set, cutoff = keys

        crossings = int(group["crossings"].sum())
        alerts = int(group["alerts"].sum())
        matched = int(group["matched_crossings"].sum())
        missed = int(group["missed_crossings"].sum())
        unmatched = int(group["unmatched_alerts"].sum())

        pooled_matches = matches_df[
            (matches_df["threshold"] == signal_threshold)
            & (matches_df["model"] == model)
            & (matches_df["feature_set"] == feature_set)
            & (matches_df["probability_cutoff"] == cutoff)
            & (matches_df["alert_outcome"] == "matched")
        ]

        pooled_rows.append({
            "threshold": signal_threshold,
            "model": model,
            "feature_set": feature_set,
            "probability_cutoff": cutoff,
            "crossings": crossings,
            "alerts": alerts,
            "matched_crossings": matched,
            "missed_crossings": missed,
            "unmatched_alerts": unmatched,
            "crossing_coverage": safe_div(matched, crossings),
            "alert_precision": safe_div(matched, alerts),
            "unmatched_alert_fraction": safe_div(unmatched, alerts),
            "median_warning_sec": (
                pooled_matches["warning_lead_sec"].median()
                if not pooled_matches.empty else np.nan
            ),
        })

    pooled = pd.DataFrame(pooled_rows)

    # Required deployment time for every hypothetical scenario.
    scenarios = list(itertools.product(
        RELAY_DISTANCES_M,
        RELAY_SPEEDS_MPS,
        SETUP_TIMES_SEC,
        RECONNECTION_TIMES_SEC,
        SAFETY_MARGINS_SEC,
    ))

    feasibility_rows = []

    for row in pooled.to_dict("records"):
        for distance, speed, setup, reconnect, safety in scenarios:
            required_sec = (
                distance / speed + setup + reconnect + safety
            )

            group_matches = matches_df[
                (matches_df["threshold"] == row["threshold"])
                & (matches_df["model"] == row["model"])
                & (matches_df["feature_set"] == row["feature_set"])
                & (
                    matches_df["probability_cutoff"]
                    == row["probability_cutoff"]
                )
                & (matches_df["alert_outcome"] == "matched")
            ]

            feasible_count = int(
                (
                    group_matches["warning_lead_sec"]
                    >= required_sec
                ).sum()
            )

            feasibility_rows.append({
                **row,
                "relay_distance_m": distance,
                "relay_speed_mps": speed,
                "setup_sec": setup,
                "reconnection_sec": reconnect,
                "safety_margin_sec": safety,
                "required_deployment_sec": required_sec,
                "feasible_crossings": feasible_count,
                "dispatch_feasible_percent_all_crossings": (
                    100 * safe_div(
                        feasible_count, row["crossings"]
                    )
                ),
            })

    feasibility = pd.DataFrame(feasibility_rows)

    pooled.to_csv(
        OUTPUT_DIR / "policy_pooled_summary.csv", index=False
    )
    feasibility.to_csv(
        OUTPUT_DIR / "policy_dispatch_feasibility.csv", index=False
    )

    # Print a compact comparison for human inspection.
    print("\n" + "=" * 72)
    print("POOLED POLICY COMPARISON")
    print("=" * 72)

    display_cols = [
        "threshold",
        "model",
        "feature_set",
        "probability_cutoff",
        "crossings",
        "alerts",
        "crossing_coverage",
        "alert_precision",
        "median_warning_sec",
    ]

    print(
        pooled[display_cols]
        .sort_values(
            ["threshold", "feature_set", "model", "probability_cutoff"]
        )
        .round(3)
        .to_string(index=False)
    )

    # Show the top candidates by feasible crossing coverage, while
    # retaining their alert counts and precision for interpretation.
    print("\n" + "=" * 72)
    print("TOP 15 DISPATCH SCENARIOS")
    print("=" * 72)

    top = feasibility.sort_values(
        [
            "dispatch_feasible_percent_all_crossings",
            "alert_precision",
            "crossing_coverage",
        ],
        ascending=False,
        na_position="last",
    ).head(15)

    top_cols = [
        "threshold",
        "model",
        "feature_set",
        "probability_cutoff",
        "relay_distance_m",
        "relay_speed_mps",
        "required_deployment_sec",
        "crossings",
        "alerts",
        "dispatch_feasible_percent_all_crossings",
        "alert_precision",
    ]

    print(top[top_cols].round(3).to_string(index=False))

    print("\nSaved results to:")
    print(OUTPUT_DIR)
    print(" - policy_by_recording.csv")
    print(" - policy_event_matches.csv")
    print(" - policy_pooled_summary.csv")
    print(" - policy_dispatch_feasibility.csv")


if __name__ == "__main__":
    main()
