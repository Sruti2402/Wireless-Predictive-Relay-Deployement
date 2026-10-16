
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = BASE_DIR / "data" / "processed" / "future_risk_labels.csv"
OUTPUT_FILE = (
    BASE_DIR / "data" / "processed" / "future_risk_labels_with_trends.csv"
)

CHANNELS = ["UL", "UR", "LL", "LR", "C"]

# Dataset sampling interval is approximately 0.206 seconds.
# These shifts approximate 1, 3, 5 and 10 seconds.
LAG_SAMPLES = {
    1: 5,
    3: 15,
    5: 24,
    10: 49,
}


def main():
    print("Loading risk-label dataset...")
    df = pd.read_csv(INPUT_FILE)

    required = [
        "recording_id",
        "elapsed_sec",
        "signal_metric",
        "hypothetical_threshold",
        "future_horizon_sec",
        "risk_label",
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # One row per timestamp is enough to calculate past-only features.
    unique = (
        df.sort_values(["recording_id", "elapsed_sec"])
        .drop_duplicates(["recording_id", "elapsed_sec"])
        .copy()
    )

    # Build features independently within each recording.
    # shift() only looks backward, so these features do not use future values.
    grouped = unique.groupby("recording_id", sort=False)

    for seconds, samples in LAG_SAMPLES.items():
        unique[f"metric_change_{seconds}s"] = (
            grouped["signal_metric"].transform(
                lambda series: series - series.shift(samples)
            )
        )

        for channel in CHANNELS:
            col = f"{channel}_level_smoothed"

            if col not in unique.columns:
                continue

            unique[f"{channel}_change_{seconds}s"] = (
                grouped[col].transform(
                    lambda series: series - series.shift(samples)
                )
            )

    feature_cols = [
        col for col in unique.columns
        if "_change_" in col
    ]

    # Merge features back to all threshold/horizon labels.
    # Many labels share the same timestamp; this is intentional.
    df = df.merge(
        unique[["recording_id", "elapsed_sec"] + feature_cols],
        on=["recording_id", "elapsed_sec"],
        how="left",
        validate="many_to_one",
    )

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nTREND FEATURE BUILD COMPLETE")
    print("=" * 60)
    print(f"Original labelled rows: {len(df):,}")
    print(f"Recordings: {df['recording_id'].nunique()}")
    print(f"Trend features created: {len(feature_cols)}")
    print(f"Output: {OUTPUT_FILE}")

    print("\nMissing-value percentage by trend feature:")
    print(
        (df[feature_cols].isna().mean() * 100)
        .round(2)
        .to_string()
    )

    print("\nAverage metric change by recording:")
    summary = (
        df.drop_duplicates(["recording_id", "elapsed_sec"])
        .groupby("recording_id")[
            ["metric_change_1s", "metric_change_3s",
             "metric_change_5s", "metric_change_10s"]
        ]
        .agg(["median", "mean"])
        .round(2)
    )
    print(summary.to_string())

    print(
        "\nNote: positive change means the metric increased; "
        "negative change means it decreased."
    )
    print(
        "These are signal-trend features, not proof of actual "
        "communication failure."
    )


if __name__ == "__main__":
    main()
