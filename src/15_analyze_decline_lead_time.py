
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "combined_repaired.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

EVENT_OUTPUT = OUTPUT_DIR / "decline_lead_time_events.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "decline_lead_time_summary.csv"
COMPARISON_OUTPUT = OUTPUT_DIR / "decline_vs_control_comparison.csv"

SIGNAL_COLUMNS = [
    "UL_level",
    "UR_level",
    "LL_level",
    "LR_level",
    "C_level",
]

HORIZONS_SECONDS = [1, 3, 5, 10, 15]

# Exploratory detection settings, not physical failure thresholds.
SMOOTHING_SECONDS = 5.0
LOOKBACK_SECONDS = 5.0
DECLINE_UNITS = 10.0
MIN_DECLINING_CHANNELS = 3

# Random comparison points per recording.
RANDOM_CONTROLS_PER_RECORDING = 100
RANDOM_SEED = 42

# Avoid sampling controls too close to a detected decline start.
CONTROL_EXCLUSION_SECONDS = 15.0


# ============================================================
# HELPERS
# ============================================================

def find_nearest_index(times, target_time):
    """Return the row index nearest to a target elapsed time."""
    position = np.searchsorted(times, target_time)

    if position <= 0:
        return 0

    if position >= len(times):
        return len(times) - 1

    previous = position - 1

    if abs(times[previous] - target_time) <= abs(
        times[position] - target_time
    ):
        return previous

    return position


def summarize_outcome(
    group,
    smoothed,
    event_index,
    baseline_index,
    event_type,
):
    """
    Measure future signal behavior relative to a pre-event baseline.

    event_index:
        Row where the decline condition first becomes true.

    baseline_index:
        Row approximately LOOKBACK_SECONDS before the event.

    This is signal dynamics analysis, not link-failure prediction.
    """
    times = group["elapsed_sec"].to_numpy(dtype=float)

    event_time = times[event_index]
    baseline_time = times[baseline_index]

    baseline = smoothed.iloc[baseline_index].to_numpy(dtype=float)

    record = {
        "recording_id": group["recording_id"].iloc[0],
        "event_type": event_type,
        "event_time_sec": event_time,
        "baseline_time_sec": baseline_time,
    }

    # Record signal values at event time and the baseline.
    for column in SIGNAL_COLUMNS:
        record[f"{column}_baseline"] = smoothed.iloc[
            baseline_index
        ][column]

        record[f"{column}_at_event"] = smoothed.iloc[
            event_index
        ][column]

    # For each horizon, calculate future endpoint and minimum.
    for horizon in HORIZONS_SECONDS:
        target_time = event_time + horizon

        # Do not evaluate horizons beyond the recording.
        if target_time > times[-1]:
            record[f"valid_{horizon}s"] = False

            for column in SIGNAL_COLUMNS:
                record[f"{column}_at_{horizon}s"] = np.nan
                record[f"{column}_delta_{horizon}s"] = np.nan
                record[f"{column}_min_{horizon}s"] = np.nan
                record[f"{column}_min_delta_{horizon}s"] = np.nan

            continue

        endpoint_index = find_nearest_index(times, target_time)

        # Include event row through the endpoint in the future window.
        future = smoothed.iloc[
            event_index:endpoint_index + 1
        ]

        record[f"valid_{horizon}s"] = True

        endpoint_values = smoothed.iloc[
            endpoint_index
        ].to_numpy(dtype=float)

        future_minimum = future.min().to_numpy(dtype=float)

        for channel_index, column in enumerate(SIGNAL_COLUMNS):
            record[f"{column}_at_{horizon}s"] = (
                endpoint_values[channel_index]
            )

            record[f"{column}_delta_{horizon}s"] = (
                endpoint_values[channel_index] - baseline[channel_index]
            )

            record[f"{column}_min_{horizon}s"] = (
                future_minimum[channel_index]
            )

            record[f"{column}_min_delta_{horizon}s"] = (
                future_minimum[channel_index] - baseline[channel_index]
            )

        # Average across channels for a compact overall measure.
        record[f"mean_delta_{horizon}s"] = float(
            np.mean(endpoint_values - baseline)
        )

        record[f"mean_min_delta_{horizon}s"] = float(
            np.mean(future_minimum - baseline)
        )

    return record


# ============================================================
# LOAD AND VALIDATE DATA
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "recording_id",
        "elapsed_sec",
        *SIGNAL_COLUMNS,
    ]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )

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

    print("=" * 75)
    print("SIGNAL DECLINE LEAD-TIME ANALYSIS")
    print("=" * 75)
    print(f"Input rows: {len(df):,}")
    print(f"Recordings: {df['recording_id'].nunique()}")
    print(f"Horizons: {HORIZONS_SECONDS} seconds")
    print(
        f"Detection rule: at least {MIN_DECLINING_CHANNELS} of "
        f"{len(SIGNAL_COLUMNS)} channels decline by at least "
        f"{DECLINE_UNITS} units over approximately "
        f"{LOOKBACK_SECONDS} seconds."
    )

    rng = np.random.default_rng(RANDOM_SEED)

    event_records = []
    recording_summaries = []

    # ========================================================
    # ANALYZE EACH RECORDING INDEPENDENTLY
    # ========================================================

    for recording_id, original_group in df.groupby(
        "recording_id", sort=True
    ):
        group = original_group.sort_values(
            "elapsed_sec"
        ).reset_index(drop=True)

        times = group["elapsed_sec"].to_numpy(dtype=float)

        if len(group) < 10:
            print(f"Skipping {recording_id}: too few rows.")
            continue

        intervals = np.diff(times)
        positive_intervals = intervals[intervals > 0]

        if len(positive_intervals) == 0:
            print(f"Skipping {recording_id}: invalid timestamps.")
            continue

        median_interval = float(np.median(positive_intervals))

        if median_interval <= 0:
            print(f"Skipping {recording_id}: invalid sample interval.")
            continue

        smoothing_samples = max(
            2, round(SMOOTHING_SECONDS / median_interval)
        )

        lookback_samples = max(
            1, round(LOOKBACK_SECONDS / median_interval)
        )

        print(
            f"DEBUG {recording_id}: "
            f"median_interval={median_interval:.4f}, "
            f"smoothing_samples={smoothing_samples}, "
            f"lookback_samples={lookback_samples}"
        )

        # Trailing rolling averages; no future samples are used
        # to compute a value at the current row.
        smoothed = group[SIGNAL_COLUMNS].rolling(
            window=smoothing_samples,
            min_periods=max(2, int(smoothing_samples * 0.8)),
        ).mean()

        # Change over approximately LOOKBACK_SECONDS.
        changes = smoothed - smoothed.shift(lookback_samples)

        channel_decline = changes <= -DECLINE_UNITS

        declining_channel_count = channel_decline.sum(axis=1)

        joint_decline = (
            declining_channel_count >= MIN_DECLINING_CHANNELS
        ).fillna(False).to_numpy(dtype=bool)

        # A decline event is the START of a joint-decline episode,
        # not every row within an episode.
        previous_joint = np.r_[False, joint_decline[:-1]]

        episode_start_indices = np.flatnonzero(
            joint_decline & ~previous_joint
        )

        print(
            f"DEBUG {recording_id}: "
            f"joint_decline_rows={int(joint_decline.sum())}, "
            f"episode_starts={len(episode_start_indices)}"
        )

        # The baseline is approximately LOOKBACK_SECONDS before
        # the event's detected start.
        valid_starts = []

        for event_index in episode_start_indices:
            baseline_time = (
                times[event_index] - LOOKBACK_SECONDS
            )

            if baseline_time < times[0]:
                continue

            baseline_index = find_nearest_index(
                times, baseline_time
            )

            if baseline_index >= event_index:
                continue

            # Require all signals at baseline and event to be valid.
            if (
                smoothed.iloc[baseline_index].isna().any()
                or smoothed.iloc[event_index].isna().any()
            ):
                continue

            valid_starts.append(int(event_index))

        # Record the detected decline episodes.
        for event_index in valid_starts:
            baseline_time = (
                times[event_index] - LOOKBACK_SECONDS
            )

            baseline_index = find_nearest_index(
                times, baseline_time
            )

            record = summarize_outcome(
                group=group,
                smoothed=smoothed,
                event_index=event_index,
                baseline_index=baseline_index,
                event_type="joint_decline",
            )

            event_records.append(record)

        # ====================================================
        # RANDOM CONTROL POINTS
        # ====================================================

        # Controls must have:
        # 1. A usable baseline.
        # 2. At least 15 seconds of future data.
        # 3. No joint-decline condition at the sampled row.
        # 4. No detected decline start within the exclusion window.
        #
        # This comparison is descriptive, not a causal experiment.

        decline_start_times = times[valid_starts]

        candidate_indices = []

        first_usable_index = smoothing_samples + lookback_samples

        for index in range(first_usable_index, len(group)):
            if times[index] - times[0] < LOOKBACK_SECONDS:
                continue

            if times[index] + max(HORIZONS_SECONDS) > times[-1]:
                continue

            if joint_decline[index]:
                continue

            if (
                smoothed.iloc[index].isna().any()
                or smoothed.iloc[index - lookback_samples].isna().any()
            ):
                continue

            if len(decline_start_times):
                distances = np.abs(
                    decline_start_times - times[index]
                )

                if np.any(
                    distances <= CONTROL_EXCLUSION_SECONDS
                ):
                    continue

            candidate_indices.append(index)

        number_of_controls = min(
            RANDOM_CONTROLS_PER_RECORDING,
            len(candidate_indices),
        )

        if number_of_controls:
            selected_controls = rng.choice(
                candidate_indices,
                size=number_of_controls,
                replace=False,
            )

            for event_index in sorted(selected_controls):
                baseline_index = event_index - lookback_samples

                record = summarize_outcome(
                    group=group,
                    smoothed=smoothed,
                    event_index=int(event_index),
                    baseline_index=int(baseline_index),
                    event_type="random_control",
                )

                event_records.append(record)

        recording_summaries.append({
            "recording_id": recording_id,
            "rows": len(group),
            "duration_sec": round(
                float(times[-1] - times[0]), 3
            ),
            "median_sample_interval_sec": round(
                median_interval, 4
            ),
            "smoothing_samples": smoothing_samples,
            "lookback_samples": lookback_samples,
            "joint_decline_rows": int(joint_decline.sum()),
            "joint_decline_episodes": len(valid_starts),
            "eligible_random_controls": len(candidate_indices),
            "selected_random_controls": number_of_controls,
        })

        print(
            f"{recording_id}: "
            f"{len(valid_starts)} decline episodes, "
            f"{number_of_controls} controls"
        )

    # ========================================================
    # SAVE EVENT-LEVEL RESULTS
    # ========================================================

    if not event_records:
        raise RuntimeError(
            "No usable events or control points were generated. "
            "Check timestamps, signal columns, and smoothing settings."
        )

    event_df = pd.DataFrame(event_records)

    event_df.to_csv(EVENT_OUTPUT, index=False)

    recording_summary_df = pd.DataFrame(recording_summaries)

    recording_summary_df.to_csv(
        SUMMARY_OUTPUT, index=False
    )

    # ========================================================
    # COMPARE DECLINE EVENTS AGAINST CONTROLS
    # ========================================================

    comparison_records = []

    for horizon in HORIZONS_SECONDS:
        delta_column = f"mean_delta_{horizon}s"
        minimum_column = f"mean_min_delta_{horizon}s"

        for event_type in ["joint_decline", "random_control"]:
            subset = event_df[
                (event_df["event_type"] == event_type)
                & event_df[f"valid_{horizon}s"]
            ]

            comparison_records.append({
                "horizon_sec": horizon,
                "event_type": event_type,
                "sample_count": len(subset),
                "mean_endpoint_delta": (
                    subset[delta_column].mean()
                    if len(subset) else np.nan
                ),
                "median_endpoint_delta": (
                    subset[delta_column].median()
                    if len(subset) else np.nan
                ),
                "mean_future_minimum_delta": (
                    subset[minimum_column].mean()
                    if len(subset) else np.nan
                ),
                "median_future_minimum_delta": (
                    subset[minimum_column].median()
                    if len(subset) else np.nan
                ),
                "percent_with_negative_endpoint_delta": (
                    100 * (
                        subset[delta_column] < 0
                    ).mean()
                    if len(subset) else np.nan
                ),
            })

    comparison_df = pd.DataFrame(comparison_records)

    comparison_df.to_csv(
        COMPARISON_OUTPUT, index=False
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n" + "=" * 75)
    print("DECLINE EPISODE SUMMARY")
    print("=" * 75)

    print(
        recording_summary_df.to_string(index=False)
    )

    print("\n" + "=" * 75)
    print("DECLINE EVENTS VS RANDOM CONTROLS")
    print("=" * 75)

    display_columns = [
        "horizon_sec",
        "event_type",
        "sample_count",
        "mean_endpoint_delta",
        "median_endpoint_delta",
        "mean_future_minimum_delta",
        "percent_with_negative_endpoint_delta",
    ]

    print(
        comparison_df[display_columns].round(3).to_string(
            index=False
        )
    )

    print("\nSaved files:")
    print(EVENT_OUTPUT)
    print(SUMMARY_OUTPUT)
    print(COMPARISON_OUTPUT)

    print("\nIMPORTANT INTERPRETATION:")
    print(
        "- Negative deltas mean signals were lower than their "
        "pre-event baseline."
    )
    print(
        "- Future minimum deltas measure the lowest smoothed "
        "value reached within each horizon."
    )
    print(
        "- These results do not establish communication failure, "
        "a failure threshold, or reliable dispatch lead time."
    )
    print(
        "- Random controls are comparison samples, not independent "
        "experimental observations. Events within a recording may "
        "also be correlated."
    )


if __name__ == "__main__":
    main()
