from pathlib import Path
import itertools

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

SIGNAL_FILE = PROCESSED / "combined_repaired.csv"
PREDICTION_FILE = PROCESSED / "trend_model_oof_predictions.csv"

# Must match the settings used in the earlier scripts.
SMOOTHING_WINDOW = 24
SMOOTHING_MIN_PERIODS = 12

PREDICTION_HORIZON_SEC = 30.0
PROBABILITY_THRESHOLD = 0.50

# Suppress repeated alerts and merge closely spaced crossings.
ALERT_COOLDOWN_SEC = 30.0
CROSSING_COOLDOWN_SEC = 30.0

SIGNAL_THRESHOLDS = [20, 30, 40]

# Hypothetical relay deployment scenarios.
RELAY_DISTANCES_M = [5, 10, 15]
RELAY_SPEEDS_MPS = [0.1, 0.2, 0.4]
SETUP_TIMES_SEC = [2, 5]
RECONNECTION_TIMES_SEC = [1, 3]
SAFETY_MARGINS_SEC = [0, 2]

SIGNAL_COLUMNS = [
    "UL_level",
    "UR_level",
    "LL_level",
    "LR_level",
    "C_level",
]


# ============================================================
# LOAD AND VALIDATE INPUTS
# ============================================================

def load_inputs():
    if not SIGNAL_FILE.exists():
        raise FileNotFoundError(
            f"Missing signal file: {SIGNAL_FILE}"
        )

    if not PREDICTION_FILE.exists():
        raise FileNotFoundError(
            f"Missing prediction file: {PREDICTION_FILE}"
        )

    signals = pd.read_csv(SIGNAL_FILE)
    predictions = pd.read_csv(PREDICTION_FILE)

    required_signal = {
        "recording_id",
        "elapsed_sec",
        *SIGNAL_COLUMNS,
    }

    required_predictions = {
        "recording_id",
        "elapsed_sec",
        "threshold",
        "horizon_sec",
        "model",
        "feature_set",
        "risk_probability",
    }

    missing_signal = required_signal - set(signals.columns)
    missing_predictions = (
        required_predictions - set(predictions.columns)
    )

    if missing_signal:
        raise ValueError(
            f"Signal file is missing columns: {sorted(missing_signal)}"
        )

    if missing_predictions:
        raise ValueError(
            "Prediction file is missing columns: "
            f"{sorted(missing_predictions)}"
        )

    signals["elapsed_sec"] = pd.to_numeric(
        signals["elapsed_sec"], errors="coerce"
    )

    predictions["elapsed_sec"] = pd.to_numeric(
        predictions["elapsed_sec"], errors="coerce"
    )

    predictions["threshold"] = pd.to_numeric(
        predictions["threshold"], errors="coerce"
    )

    predictions["horizon_sec"] = pd.to_numeric(
        predictions["horizon_sec"], errors="coerce"
    )

    predictions["risk_probability"] = pd.to_numeric(
        predictions["risk_probability"], errors="coerce"
    )

    signals = signals.dropna(
        subset=["recording_id", "elapsed_sec"]
    ).copy()

    predictions = predictions.dropna(
        subset=[
            "recording_id",
            "elapsed_sec",
            "threshold",
            "horizon_sec",
            "model",
            "feature_set",
            "risk_probability",
        ]
    ).copy()

    predictions["risk_probability"] = predictions[
        "risk_probability"
    ].clip(0, 1)

    predictions = predictions[
        np.isclose(
            predictions["horizon_sec"],
            PREDICTION_HORIZON_SEC,
        )
        & predictions["threshold"].isin(SIGNAL_THRESHOLDS)
    ].copy()

    if predictions.empty:
        raise ValueError(
            "No predictions found for the configured thresholds "
            "and 30-second horizon. Check the CSV values."
        )

    return signals, predictions


# ============================================================
# RECONSTRUCT THE OBSERVED SMOOTHED SIGNAL METRIC
# ============================================================

def build_signal_metric(signals):
    pieces = []

    for recording_id, group in signals.groupby(
        "recording_id", sort=False
    ):
        group = group.sort_values("elapsed_sec").copy()

        # Coerce signal columns to numeric.
        for column in SIGNAL_COLUMNS:
            group[column] = pd.to_numeric(
                group[column], errors="coerce"
            )

        # Smooth each channel independently.
        smoothed = pd.DataFrame(index=group.index)

        for column in SIGNAL_COLUMNS:
            smoothed[column] = group[column].rolling(
                window=SMOOTHING_WINDOW,
                min_periods=SMOOTHING_MIN_PERIODS,
            ).mean()

        # Median across the five smoothed signal channels.
        # Pandas median ignores missing channel values.
        group["signal_metric"] = smoothed.median(
            axis=1, skipna=True
        )

        pieces.append(
            group[
                [
                    "recording_id",
                    "elapsed_sec",
                    "signal_metric",
                ]
            ]
        )

    result = pd.concat(pieces, ignore_index=True)

    return result.sort_values(
        ["recording_id", "elapsed_sec"]
    ).reset_index(drop=True)


# ============================================================
# FIND OBSERVED DOWNWARD THRESHOLD CROSSINGS
# ============================================================

def detect_crossings(metric_data):
    events = []

    for threshold in SIGNAL_THRESHOLDS:
        for recording_id, group in metric_data.groupby(
            "recording_id", sort=False
        ):
            group = group.sort_values("elapsed_sec").copy()

            metric = group["signal_metric"]
            previous_metric = metric.shift(1)

            crossing_mask = (
                metric.notna()
                & previous_metric.notna()
                & (previous_metric > threshold)
                & (metric <= threshold)
            )

            candidates = group.loc[
                crossing_mask,
                ["elapsed_sec", "signal_metric"],
            ]

            # Merge repeated crossings that happen close together.
            # Keep the first crossing in each 30-second window.
            last_kept_time = -np.inf

            for row in candidates.itertuples(index=False):
                crossing_time = float(row.elapsed_sec)

                if (
                    crossing_time - last_kept_time
                    < CROSSING_COOLDOWN_SEC
                ):
                    continue

                events.append(
                    {
                        "recording_id": recording_id,
                        "threshold": threshold,
                        "crossing_time_sec": crossing_time,
                        "metric_at_crossing": float(
                            row.signal_metric
                        ),
                    }
                )

                last_kept_time = crossing_time

    return pd.DataFrame(
        events,
        columns=[
            "recording_id",
            "threshold",
            "crossing_time_sec",
            "metric_at_crossing",
        ],
    )


# ============================================================
# CONVERT PROBABILITIES INTO ALERT EPISODES
# ============================================================

def generate_alerts(predictions):
    alerts = []

    group_columns = [
        "threshold",
        "model",
        "feature_set",
        "recording_id",
    ]

    for keys, group in predictions.groupby(
        group_columns, sort=False
    ):
        threshold, model, feature_set, recording_id = keys

        group = group.sort_values("elapsed_sec").copy()

        group["above_probability_threshold"] = (
            group["risk_probability"]
            >= PROBABILITY_THRESHOLD
        )

        # Alert only on a transition from below to above the
        # probability threshold, rather than on every high row.
        group["previously_above"] = (
            group["above_probability_threshold"]
            .shift(1, fill_value=False)
        )

        rising_edges = group[
            group["above_probability_threshold"]
            & ~group["previously_above"]
        ]

        last_alert_time = -np.inf

        for row in rising_edges.itertuples(index=False):
            alert_time = float(row.elapsed_sec)

            if (
                alert_time - last_alert_time
                < ALERT_COOLDOWN_SEC
            ):
                continue

            alerts.append(
                {
                    "threshold": int(threshold),
                    "model": model,
                    "feature_set": feature_set,
                    "recording_id": recording_id,
                    "alert_time_sec": alert_time,
                    "risk_probability": float(
                        row.risk_probability
                    ),
                }
            )

            last_alert_time = alert_time

    return pd.DataFrame(
        alerts,
        columns=[
            "threshold",
            "model",
            "feature_set",
            "recording_id",
            "alert_time_sec",
            "risk_probability",
        ],
    )


# ============================================================
# MATCH ALERTS TO FUTURE CROSSING EVENTS
# ============================================================

def match_alerts_to_crossings(alerts, crossings):
    """
    Each alert can match at most one future crossing.
    Each crossing can receive at most one matched alert.

    An unmatched alert is false relative to the selected
    30-second hypothetical crossing horizon.

    A crossing without a matched alert is a missed warning.
    """

    matched_alert_rows = []

    if alerts.empty:
        return pd.DataFrame(
            columns=[
                *alerts.columns,
                "matched_crossing_time_sec",
                "warning_lead_sec",
                "alert_outcome",
            ]
        )

    model_groups = [
        "threshold",
        "model",
        "feature_set",
        "recording_id",
    ]

    for keys, alert_group in alerts.groupby(
        model_groups, sort=False
    ):
        threshold, model, feature_set, recording_id = keys

        event_group = crossings[
            (crossings["threshold"] == threshold)
            & (crossings["recording_id"] == recording_id)
        ].sort_values("crossing_time_sec")

        event_times = event_group[
            "crossing_time_sec"
        ].to_numpy(dtype=float)

        matched_event_indices = set()

        for alert in alert_group.sort_values(
            "alert_time_sec"
        ).itertuples(index=False):

            alert_time = float(alert.alert_time_sec)

            # First crossing at or after this alert.
            candidate_indices = np.where(
                (event_times >= alert_time)
                & (
                    event_times
                    <= alert_time + PREDICTION_HORIZON_SEC
                )
            )[0]

            matched_index = None

            for index in candidate_indices:
                if index not in matched_event_indices:
                    matched_index = int(index)
                    break

            row = {
                "threshold": int(threshold),
                "model": model,
                "feature_set": feature_set,
                "recording_id": recording_id,
                "alert_time_sec": alert_time,
                "risk_probability": float(
                    alert.risk_probability
                ),
                "matched_crossing_time_sec": np.nan,
                "warning_lead_sec": np.nan,
                "alert_outcome": "false_alert",
            }

            if matched_index is not None:
                crossing_time = float(
                    event_times[matched_index]
                )

                matched_event_indices.add(matched_index)

                row["matched_crossing_time_sec"] = crossing_time
                row["warning_lead_sec"] = (
                    crossing_time - alert_time
                )
                row["alert_outcome"] = "matched_warning"

            matched_alert_rows.append(row)

    return pd.DataFrame(matched_alert_rows)


# ============================================================
# SUMMARIZE DETECTION AND DISPATCH FEASIBILITY
# ============================================================

def build_summaries(alert_results, crossings):
    summary_rows = []
    dispatch_rows = []

    if alert_results.empty:
        alert_results = pd.DataFrame(
            columns=[
                "threshold",
                "model",
                "feature_set",
                "recording_id",
                "alert_time_sec",
                "risk_probability",
                "matched_crossing_time_sec",
                "warning_lead_sec",
                "alert_outcome",
            ]
        )

    # Compare only model/feature combinations actually present.
    prediction_combinations = (
        alert_results[
            ["threshold", "model", "feature_set"]
        ]
        .drop_duplicates()
    )

    # Alerts may be empty for a configuration. Recover all
    # configurations from the saved prediction data where needed
    # in main(), before calling this function.
    for combo in prediction_combinations.itertuples(
        index=False
    ):
        threshold = int(combo.threshold)
        model = combo.model
        feature_set = combo.feature_set

        a = alert_results[
            (alert_results["threshold"] == threshold)
            & (alert_results["model"] == model)
            & (alert_results["feature_set"] == feature_set)
        ]

        c = crossings[
            crossings["threshold"] == threshold
        ]

        total_crossings = len(c)
        matched = a[
            a["alert_outcome"] == "matched_warning"
        ]
        false_alerts = a[
            a["alert_outcome"] == "false_alert"
        ]

        matched_crossing_count = (
            matched["matched_crossing_time_sec"]
            .nunique()
        )

        warning_leads = matched["warning_lead_sec"].dropna()

        summary_rows.append(
            {
                "threshold": threshold,
                "model": model,
                "feature_set": feature_set,
                "observed_crossings": total_crossings,
                "crossings_with_warning": matched_crossing_count,
                "crossings_without_warning": max(
                    0, total_crossings - matched_crossing_count
                ),
                "crossing_warning_coverage_percent": (
                    100 * matched_crossing_count / total_crossings
                    if total_crossings else np.nan
                ),
                "total_alerts": len(a),
                "matched_alerts": len(matched),
                "false_alerts": len(false_alerts),
                "alert_precision_percent": (
                    100 * len(matched) / len(a)
                    if len(a) else np.nan
                ),
                "median_warning_lead_sec": (
                    warning_leads.median()
                    if len(warning_leads) else np.nan
                ),
                "mean_warning_lead_sec": (
                    warning_leads.mean()
                    if len(warning_leads) else np.nan
                ),
            }
        )

        scenarios = itertools.product(
            RELAY_DISTANCES_M,
            RELAY_SPEEDS_MPS,
            SETUP_TIMES_SEC,
            RECONNECTION_TIMES_SEC,
            SAFETY_MARGINS_SEC,
        )

        for (
            distance,
            speed,
            setup,
            reconnect,
            safety,
        ) in scenarios:

            required_time = (
                distance / speed
                + setup
                + reconnect
                + safety
            )

            feasible = warning_leads[
                warning_leads >= required_time
            ]

            dispatch_rows.append(
                {
                    "threshold": threshold,
                    "model": model,
                    "feature_set": feature_set,
                    "relay_distance_m": distance,
                    "relay_speed_mps": speed,
                    "setup_time_sec": setup,
                    "reconnection_time_sec": reconnect,
                    "safety_margin_sec": safety,
                    "required_lead_time_sec": required_time,
                    "observed_crossings": total_crossings,
                    "crossings_with_warning": matched_crossing_count,
                    "dispatch_feasible_crossings": len(feasible),
                    "crossings_not_dispatch_feasible": max(
                        0, total_crossings - len(feasible)
                    ),
                    "dispatch_feasible_percent_all_crossings": (
                        100 * len(feasible) / total_crossings
                        if total_crossings else np.nan
                    ),
                    "dispatch_feasible_percent_warned_crossings": (
                        100 * len(feasible) / matched_crossing_count
                        if matched_crossing_count else np.nan
                    ),
                    "false_alerts": len(false_alerts),
                    "median_warning_lead_sec": (
                        warning_leads.median()
                        if len(warning_leads) else np.nan
                    ),
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(dispatch_rows)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 76)
    print("EVENT-BASED RELAY DISPATCH EVALUATION")
    print("=" * 76)

    signals, predictions = load_inputs()

    print(f"Signal rows: {len(signals):,}")
    print(f"Prediction rows: {len(predictions):,}")
    print(
        "Recordings:",
        signals["recording_id"].nunique(),
    )
    print(
        "Model configurations:",
        predictions[
            ["threshold", "model", "feature_set"]
        ].drop_duplicates().shape[0],
    )

    metric_data = build_signal_metric(signals)
    crossings = detect_crossings(metric_data)
    alerts = generate_alerts(predictions)

    print(f"Observed downward crossings: {len(crossings):,}")
    print(f"Alert episodes: {len(alerts):,}")

    # Save intermediate event tables so results are inspectable.
    metric_data.to_csv(
        PROCESSED / "observed_signal_metric.csv",
        index=False,
    )

    crossings.to_csv(
        PROCESSED / "observed_threshold_crossings.csv",
        index=False,
    )

    alerts.to_csv(
        PROCESSED / "relay_alert_episodes.csv",
        index=False,
    )

    alert_results = match_alerts_to_crossings(
        alerts, crossings
    )

    alert_results.to_csv(
        PROCESSED / "relay_alert_event_matches.csv",
        index=False,
    )

    # Build summaries for all prediction configurations, even
    # configurations that produced zero alert episodes.
    combinations = predictions[
        ["threshold", "model", "feature_set"]
    ].drop_duplicates()

    summary_rows = []
    dispatch_rows = []

    for combo in combinations.itertuples(index=False):
        threshold = int(combo.threshold)
        model = combo.model
        feature_set = combo.feature_set

        a = alert_results[
            (alert_results["threshold"] == threshold)
            & (alert_results["model"] == model)
            & (alert_results["feature_set"] == feature_set)
        ]

        c = crossings[
            crossings["threshold"] == threshold
        ]

        matched = a[
            a["alert_outcome"] == "matched_warning"
        ]

        false_alerts = a[
            a["alert_outcome"] == "false_alert"
        ]

        leads = matched["warning_lead_sec"].dropna()
        total_crossings = len(c)
        warned_crossings = matched[
            "matched_crossing_time_sec"
        ].nunique()

        summary_rows.append(
            {
                "threshold": threshold,
                "model": model,
                "feature_set": feature_set,
                "observed_crossings": total_crossings,
                "crossings_with_warning": warned_crossings,
                "crossings_without_warning": max(
                    0, total_crossings - warned_crossings
                ),
                "crossing_warning_coverage_percent": (
                    100 * warned_crossings / total_crossings
                    if total_crossings else np.nan
                ),
                "total_alerts": len(a),
                "matched_alerts": len(matched),
                "false_alerts": len(false_alerts),
                "alert_precision_percent": (
                    100 * len(matched) / len(a)
                    if len(a) else np.nan
                ),
                "median_warning_lead_sec": (
                    leads.median() if len(leads) else np.nan
                ),
                "mean_warning_lead_sec": (
                    leads.mean() if len(leads) else np.nan
                ),
            }
        )

        for (
            distance,
            speed,
            setup,
            reconnect,
            safety,
        ) in itertools.product(
            RELAY_DISTANCES_M,
            RELAY_SPEEDS_MPS,
            SETUP_TIMES_SEC,
            RECONNECTION_TIMES_SEC,
            SAFETY_MARGINS_SEC,
        ):
            required_time = (
                distance / speed + setup + reconnect + safety
            )

            feasible_count = int(
                (leads >= required_time).sum()
            )

            dispatch_rows.append(
                {
                    "threshold": threshold,
                    "model": model,
                    "feature_set": feature_set,
                    "relay_distance_m": distance,
                    "relay_speed_mps": speed,
                    "setup_time_sec": setup,
                    "reconnection_time_sec": reconnect,
                    "safety_margin_sec": safety,
                    "required_lead_time_sec": required_time,
                    "observed_crossings": total_crossings,
                    "crossings_with_warning": warned_crossings,
                    "dispatch_feasible_crossings": feasible_count,
                    "crossings_not_dispatch_feasible": max(
                        0, total_crossings - feasible_count
                    ),
                    "dispatch_feasible_percent_all_crossings": (
                        100 * feasible_count / total_crossings
                        if total_crossings else np.nan
                    ),
                    "dispatch_feasible_percent_warned_crossings": (
                        100 * feasible_count / warned_crossings
                        if warned_crossings else np.nan
                    ),
                    "false_alerts": len(false_alerts),
                    "median_warning_lead_sec": (
                        leads.median() if len(leads) else np.nan
                    ),
                }
            )

    summary = pd.DataFrame(summary_rows)
    dispatch = pd.DataFrame(dispatch_rows)

    summary_path = (
        PROCESSED / "relay_event_evaluation_summary.csv"
    )
    dispatch_path = (
        PROCESSED / "relay_dispatch_feasibility.csv"
    )

    summary.to_csv(summary_path, index=False)
    dispatch.to_csv(dispatch_path, index=False)

    print("\n" + "=" * 76)
    print("ALERT AND CROSSING SUMMARY")
    print("=" * 76)

    if not summary.empty:
        print(
            summary.sort_values(
                ["threshold", "feature_set", "model"]
            ).to_string(index=False, float_format=lambda x: f"{x:.3f}")
        )

    print("\n" + "=" * 76)
    print("BEST DISPATCH SCENARIOS BY FEASIBLE CROSSING RATE")
    print("=" * 76)

    if not dispatch.empty:
        best = (
            dispatch.sort_values(
                [
                    "threshold",
                    "dispatch_feasible_percent_all_crossings",
                    "false_alerts",
                ],
                ascending=[True, False, True],
                na_position="last",
            )
            .groupby(
                ["threshold", "model", "feature_set"],
                as_index=False,
            )
            .head(3)
        )

        print(
            best.to_string(
                index=False,
                float_format=lambda x: f"{x:.3f}",
            )
        )

    print("\nSaved files:")
    for path in [
        PROCESSED / "observed_signal_metric.csv",
        PROCESSED / "observed_threshold_crossings.csv",
        PROCESSED / "relay_alert_episodes.csv",
        PROCESSED / "relay_alert_event_matches.csv",
        summary_path,
        dispatch_path,
    ]:
        print(f"  {path}")

    print("\nIMPORTANT LIMITATIONS:")
    print(
        "- Threshold crossings are defined from the smoothed "
        "median of five signal channels."
    )
    print(
        "- They are not confirmed wireless link failures."
    )
    print(
        "- Alerts are derived from out-of-fold predictions at a "
        "fixed probability threshold of 0.50."
    )
    print(
        "- Dispatch feasibility uses hypothetical relay travel, "
        "setup, reconnection, and safety times."
    )
    print(
        "- This is retrospective evaluation, not a live deployment "
        "or proof of causal advance warning."
    )


if __name__ == "__main__":
    main()