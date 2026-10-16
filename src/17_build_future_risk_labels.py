
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "combined_repaired.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_DATASET = OUTPUT_DIR / "future_risk_labels.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "future_risk_label_summary.csv"

# Main signal channels. These are the channels observed in our dataset.
SIGNAL_COLUMNS = [
    "UL_level",
    "UR_level",
    "LL_level",
    "LR_level",
    "C_level",
]

# Hypothetical signal thresholds, not confirmed failure thresholds.
SIGNAL_THRESHOLDS = [20, 30, 40, 50]

# Future time windows in seconds.
FUTURE_HORIZONS_SEC = [5, 10, 15, 30, 60, 120, 180]

# Trailing smoothing window: approximately 5 seconds at ~4.85 Hz.
SMOOTHING_SAMPLES = 24
MIN_SMOOTHING_SAMPLES = 12


# ============================================================
# BUILD DATASET
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "recording_id",
        "elapsed_sec",
        *SIGNAL_COLUMNS,
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["elapsed_sec"] = pd.to_numeric(
        df["elapsed_sec"], errors="coerce"
    )

    for column in SIGNAL_COLUMNS:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    df = df.dropna(
        subset=["recording_id", "elapsed_sec", *SIGNAL_COLUMNS]
    ).copy()

    df = df.sort_values(
        ["recording_id", "elapsed_sec"]
    ).reset_index(drop=True)

    # Avoid duplicate timestamps within a recording.
    df = df.drop_duplicates(
        subset=["recording_id", "elapsed_sec"],
        keep="first",
    ).reset_index(drop=True)

    print(f"Input rows after cleaning: {len(df):,}")
    print(f"Recordings: {df['recording_id'].nunique()}")

    all_rows = []

    # Process each recording independently to prevent data leakage
    # across recording boundaries.
    for recording_id, recording in df.groupby(
        "recording_id", sort=False
    ):
        recording = recording.sort_values(
            "elapsed_sec"
        ).reset_index(drop=True)

        times = recording["elapsed_sec"].to_numpy(dtype=float)
        signal_data = recording[SIGNAL_COLUMNS].to_numpy(dtype=float)

        row_count = len(recording)

        # Smooth each channel using only present and past samples.
        smoothed = (
            recording[SIGNAL_COLUMNS]
            .rolling(
                window=SMOOTHING_SAMPLES,
                min_periods=MIN_SMOOTHING_SAMPLES,
            )
            .mean()
            .to_numpy(dtype=float)
        )

        # One combined signal-risk metric.
        metric = np.nanmedian(smoothed, axis=1)

        # Features available at the current time.
        for threshold in SIGNAL_THRESHOLDS:
            for horizon in FUTURE_HORIZONS_SEC:

                for i in range(row_count):
                    current_time = times[i]
                    current_metric = metric[i]

                    if not np.isfinite(current_metric):
                        continue

                    # Only predict future threshold crossings from
                    # points currently ABOVE the threshold.
                    # Already-breached points are not valid prediction
                    # examples for this threshold.
                    if current_metric <= threshold:
                        continue

                    horizon_end = current_time + horizon

                    # Require the recording to cover the entire
                    # future horizon. Do not label truncated tails.
                    if times[-1] < horizon_end:
                        continue

                    # Strictly future observations, through horizon_end.
                    start_index = np.searchsorted(
                        times, current_time, side="right"
                    )

                    end_index = np.searchsorted(
                        times, horizon_end, side="right"
                    )

                    future_values = metric[start_index:end_index]
                    future_values = future_values[
                        np.isfinite(future_values)
                    ]

                    if len(future_values) == 0:
                        continue

                    future_minimum = float(np.min(future_values))

                    # Binary target:
                    # 1 = threshold reached within the future window
                    # 0 = threshold not reached within that window
                    risk_label = int(future_minimum <= threshold)

                    # First crossing time, if there is one.
                    crossing_indices = np.where(
                        np.isfinite(metric[start_index:end_index])
                        & (metric[start_index:end_index] <= threshold)
                    )[0]

                    if len(crossing_indices):
                        first_crossing_index = (
                            start_index + int(crossing_indices[0])
                        )
                        time_to_crossing = float(
                            times[first_crossing_index] - current_time
                        )
                    else:
                        time_to_crossing = np.nan

                    row = {
                        "recording_id": recording_id,
                        "elapsed_sec": current_time,
                        "signal_metric": current_metric,
                        "hypothetical_threshold": threshold,
                        "future_horizon_sec": horizon,
                        "future_minimum_metric": future_minimum,
                        "risk_label": risk_label,
                        "time_to_crossing_sec": time_to_crossing,
                    }

                    # Current channel values and trailing-smoothed
                    # channel values are potential model features.
                    for j, column in enumerate(SIGNAL_COLUMNS):
                        row[column] = signal_data[i, j]
                        row[f"{column}_smoothed"] = smoothed[i, j]

                    all_rows.append(row)

        print(
            f"Processed recording {recording_id}: "
            f"{row_count:,} rows"
        )

    if not all_rows:
        raise RuntimeError(
            "No labelled rows were produced. Check the data, "
            "signal columns, and recording durations."
        )

    dataset = pd.DataFrame(all_rows)

    dataset = dataset.sort_values(
        [
            "recording_id",
            "elapsed_sec",
            "hypothetical_threshold",
            "future_horizon_sec",
        ]
    ).reset_index(drop=True)

    dataset.to_csv(OUTPUT_DATASET, index=False)

    # --------------------------------------------------------
    # LABEL SUMMARY
    # --------------------------------------------------------

    summary_rows = []

    for (threshold, horizon), subset in dataset.groupby(
        ["hypothetical_threshold", "future_horizon_sec"],
        sort=True,
    ):
        positive_count = int((subset["risk_label"] == 1).sum())
        negative_count = int((subset["risk_label"] == 0).sum())
        total = len(subset)

        summary_rows.append({
            "hypothetical_threshold": threshold,
            "future_horizon_sec": horizon,
            "total_labelled_rows": total,
            "future_crossing_count": positive_count,
            "no_crossing_count": negative_count,
            "future_crossing_percent": (
                100 * positive_count / total if total else np.nan
            ),
            "recordings_represented": subset["recording_id"].nunique(),
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_SUMMARY, index=False)

    print("\n" + "=" * 75)
    print("FUTURE-RISK LABEL DATASET CREATED")
    print("=" * 75)

    print(f"Labelled rows: {len(dataset):,}")
    print(f"Output dataset: {OUTPUT_DATASET}")
    print(f"Output summary: {OUTPUT_SUMMARY}")

    print("\nLABEL BALANCE BY THRESHOLD AND HORIZON")
    print(
        summary.to_string(
            index=False,
            formatters={
                "future_crossing_percent": "{:.2f}%".format,
            },
        )
    )

    print("\nIMPORTANT INTERPRETATION")
    print(
        "A positive label means the smoothed median signal metric "
        "reached a hypothetical threshold within the future horizon."
    )
    print(
        "It does NOT establish that communication actually failed."
    )
    print(
        "Use recording-level train/test splits when training models."
    )


if __name__ == "__main__":
    main()
