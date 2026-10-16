
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION: HYPOTHETICAL SCENARIOS, NOT MEASURED HARDWARE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "combined_repaired.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

EVENT_OUTPUT = OUTPUT_DIR / "relay_dispatch_scenarios.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "relay_dispatch_summary.csv"

SIGNAL_COLUMNS = [
    "UL_level",
    "UR_level",
    "LL_level",
    "LR_level",
    "C_level",
]

# Illustrative thresholds only. These are NOT verified failure limits.
SIGNAL_THRESHOLDS = [20, 30, 40, 50]

# Assumed distance from the relay's starting point to its deployment point.
RELAY_TRAVEL_DISTANCES_M = [5, 10, 15]

# Assumed relay movement speeds in metres/second.
RELAY_SPEEDS_MPS = [0.1, 0.2, 0.4]

# Assumed setup and reconnection durations in seconds.
SETUP_TIMES_SEC = [2, 5]
RECONNECTION_TIMES_SEC = [1, 3]

# Additional time reserved for uncertainty.
SAFETY_MARGINS_SEC = [0, 2]

# Signal decline detector, matching the previous analysis.
SMOOTHING_SECONDS = 5.0
LOOKBACK_SECONDS = 5.0
DECLINE_UNITS = 10.0
MIN_DECLINING_CHANNELS = 3


# ============================================================
# HELPERS
# ============================================================

def nearest_index(times, target_time):
    """Return the row nearest to target_time."""
    pos = int(np.searchsorted(times, target_time))

    if pos <= 0:
        return 0

    if pos >= len(times):
        return len(times) - 1

    if abs(times[pos - 1] - target_time) <= abs(
        times[pos] - target_time
    ):
        return pos - 1

    return pos


def detect_decline_episodes(group, median_interval):
    """
    Detect the start of each joint-decline episode.

    A decline is detected when at least three of five smoothed
    channels fall by at least 10 units over approximately five
    seconds.

    This detects signal deterioration, NOT communication failure.
    """
    smoothing_samples = max(
        2, round(SMOOTHING_SECONDS / median_interval)
    )
    lookback_samples = max(
        1, round(LOOKBACK_SECONDS / median_interval)
    )

    smoothed = group[SIGNAL_COLUMNS].rolling(
        window=smoothing_samples,
        min_periods=max(2, int(smoothing_samples * 0.8)),
    ).mean()

    changes = smoothed - smoothed.shift(lookback_samples)

    channel_decline = changes <= -DECLINE_UNITS
    declining_count = channel_decline.sum(axis=1)

    joint_decline = (
        declining_count >= MIN_DECLINING_CHANNELS
    ).fillna(False).to_numpy(dtype=bool)

    previous_joint = np.r_[False, joint_decline[:-1]]

    starts = np.flatnonzero(joint_decline & ~previous_joint)

    return smoothed, starts, smoothing_samples, lookback_samples


def get_threshold_timing(times, metric, event_index, threshold):
    """
    Estimate signal-threshold crossing time relative to detection.

    Positive warning:
        Crossing happens after the detector triggers.

    Negative warning:
        Threshold crossing already happened before detection.

    No observed crossing:
        The recording does not provide a usable crossing time.
    """
    event_time = float(times[event_index])
    event_signal = float(metric[event_index])

    if not np.isfinite(event_signal):
        return {
            "threshold_crossing_time_sec": np.nan,
            "warning_time_sec": np.nan,
            "threshold_state": "signal_unavailable",
        }

    previous = metric[:-1]
    current = metric[1:]

    crossing_indices = np.flatnonzero(
        (previous > threshold) & (current <= threshold)
    ) + 1

    prior_crossings = crossing_indices[
        times[crossing_indices] <= event_time
    ]

    future_crossings = crossing_indices[
        times[crossing_indices] > event_time
    ]

    if event_signal <= threshold:
        if len(prior_crossings):
            crossing_index = int(prior_crossings[-1])
            crossing_time = float(times[crossing_index])

            return {
                "threshold_crossing_time_sec": crossing_time,
                "warning_time_sec": crossing_time - event_time,
                "threshold_state": "already_below_at_detection",
            }

        return {
            "threshold_crossing_time_sec": np.nan,
            "warning_time_sec": np.nan,
            "threshold_state": "below_without_observed_crossing",
        }

    if len(future_crossings):
        crossing_index = int(future_crossings[0])
        crossing_time = float(times[crossing_index])

        return {
            "threshold_crossing_time_sec": crossing_time,
            "warning_time_sec": crossing_time - event_time,
            "threshold_state": "future_crossing_observed",
        }

    return {
        "threshold_crossing_time_sec": np.nan,
        "warning_time_sec": np.nan,
        "threshold_state": "no_crossing_observed",
    }


def simulate_dispatch(
    warning_time,
    threshold_state,
    distance_m,
    speed_mps,
    setup_sec,
    reconnect_sec,
    safety_margin_sec,
):
    """
    Compare estimated warning time with the required relay response.

    This is a scenario calculation, not a real hardware guarantee.
    """
    travel_time = distance_m / speed_mps

    response_time = travel_time + setup_sec + reconnect_sec

    required_time = response_time + safety_margin_sec

    if threshold_state in {
        "already_below_at_detection",
        "below_without_observed_crossing",
    }:
        outcome = "threshold_already_breached"

    elif threshold_state == "no_crossing_observed":
        outcome = "no_crossing_observed"

    elif not np.isfinite(warning_time):
        outcome = "insufficient_data"

    elif warning_time >= required_time:
        outcome = "dispatch_in_time_under_scenario"

    else:
        outcome = "dispatch_too_late_under_scenario"

    return {
        "travel_time_sec": travel_time,
        "setup_time_sec": setup_sec,
        "reconnection_time_sec": reconnect_sec,
        "relay_response_time_sec": response_time,
        "safety_margin_sec": safety_margin_sec,
        "required_warning_sec": required_time,
        "dispatch_outcome": outcome,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "recording_id",
        "elapsed_sec",
        *SIGNAL_COLUMNS,
    ]

    missing = [
        col for col in required_columns if col not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["elapsed_sec"] = pd.to_numeric(
        df["elapsed_sec"], errors="coerce"
    )

    for col in SIGNAL_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=["recording_id", "elapsed_sec", *SIGNAL_COLUMNS]
    ).copy()

    df = df.sort_values(
        ["recording_id", "elapsed_sec"]
    ).reset_index(drop=True)

    print("=" * 75)
    print("PREDICTIVE RELAY DEPLOYMENT SCENARIO SIMULATOR")
    print("=" * 75)
    print(f"Rows: {len(df):,}")
    print(f"Recordings: {df['recording_id'].nunique()}")
    print(f"Threshold scenarios: {SIGNAL_THRESHOLDS}")
    print(f"Travel distances (m): {RELAY_TRAVEL_DISTANCES_M}")
    print(f"Relay speeds (m/s): {RELAY_SPEEDS_MPS}")
    print(f"Setup times (s): {SETUP_TIMES_SEC}")
    print(f"Reconnection times (s): {RECONNECTION_TIMES_SEC}")
    print(f"Safety margins (s): {SAFETY_MARGINS_SEC}")

    event_rows = []

    # --------------------------------------------------------
    # STEP 1: FIND DECLINE EPISODES AND SIGNAL CROSSINGS
    # --------------------------------------------------------

    for recording_id, original_group in df.groupby(
        "recording_id", sort=True
    ):
        group = original_group.sort_values(
            "elapsed_sec"
        ).reset_index(drop=True)

        times = group["elapsed_sec"].to_numpy(dtype=float)

        intervals = np.diff(times)
        intervals = intervals[intervals > 0]

        if len(intervals) == 0:
            print(f"Skipping {recording_id}: invalid timestamps.")
            continue

        median_interval = float(np.median(intervals))

        smoothed, starts, smoothing_samples, lookback_samples = (
            detect_decline_episodes(group, median_interval)
        )

        # Exploratory aggregate metric: median across five channels.
        # We do not know whether these channels have identical semantics.
        metric = smoothed.median(axis=1).to_numpy(dtype=float)

        print(
            f"{recording_id}: {len(starts)} decline episodes, "
            f"sample interval={median_interval:.4f}s"
        )

        # ----------------------------------------------------
        # STEP 2: EVALUATE EACH DECLINE AGAINST EACH THRESHOLD
        # ----------------------------------------------------

        for episode_number, event_index in enumerate(starts, start=1):
            event_time = float(times[event_index])

            signal_at_detection = float(metric[event_index])

            for threshold in SIGNAL_THRESHOLDS:
                timing = get_threshold_timing(
                    times=times,
                    metric=metric,
                    event_index=int(event_index),
                    threshold=threshold,
                )

                for distance_m in RELAY_TRAVEL_DISTANCES_M:
                    for speed_mps in RELAY_SPEEDS_MPS:
                        for setup_sec in SETUP_TIMES_SEC:
                            for reconnect_sec in RECONNECTION_TIMES_SEC:
                                for margin_sec in SAFETY_MARGINS_SEC:
                                    dispatch = simulate_dispatch(
                                        warning_time=timing[
                                            "warning_time_sec"
                                        ],
                                        threshold_state=timing[
                                            "threshold_state"
                                        ],
                                        distance_m=distance_m,
                                        speed_mps=speed_mps,
                                        setup_sec=setup_sec,
                                        reconnect_sec=reconnect_sec,
                                        safety_margin_sec=margin_sec,
                                    )

                                    event_rows.append({
                                        "recording_id": recording_id,
                                        "episode_number": episode_number,
                                        "detection_time_sec": event_time,
                                        "signal_metric": (
                                            "median_of_five_smoothed_channels"
                                        ),
                                        "signal_at_detection": (
                                            signal_at_detection
                                        ),
                                        "hypothetical_threshold": threshold,
                                        **timing,
                                        "relay_distance_m": distance_m,
                                        "relay_speed_mps": speed_mps,
                                        **dispatch,
                                    })

    if not event_rows:
        raise RuntimeError(
            "No decline episodes were found. "
            "Check the input data and detector settings."
        )

    # --------------------------------------------------------
    # STEP 3: SAVE EVENT-LEVEL SCENARIOS
    # --------------------------------------------------------

    results = pd.DataFrame(event_rows)

    results.to_csv(EVENT_OUTPUT, index=False)

    # --------------------------------------------------------
    # STEP 4: SUMMARIZE DISPATCH OUTCOMES
    # --------------------------------------------------------

    grouping = [
        "hypothetical_threshold",
        "relay_distance_m",
        "relay_speed_mps",
        "setup_time_sec",
        "reconnection_time_sec",
        "safety_margin_sec",
    ]

    summary_rows = []

    for keys, subset in results.groupby(grouping, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        group_values = {
            column: subset[column].iloc[0]
            for column in grouping
        }

        counts = subset["dispatch_outcome"].value_counts()

        total = len(subset)

        row = {
            **group_values,
            "total_episode_scenarios": total,
            "dispatch_in_time_count": int(
                counts.get("dispatch_in_time_under_scenario", 0)
            ),
            "dispatch_too_late_count": int(
                counts.get("dispatch_too_late_under_scenario", 0)
            ),
            "threshold_already_breached_count": int(
                counts.get("threshold_already_breached", 0)
            ),
            "no_crossing_observed_count": int(
                counts.get("no_crossing_observed", 0)
            ),
            "insufficient_data_count": int(
                counts.get("insufficient_data", 0)
            ),
        }

        # Only episodes with a future crossing and finite warning time
        # can contribute to this conditional timing statistic.
        valid_future = subset[
            (subset["threshold_state"] == "future_crossing_observed")
            & subset["warning_time_sec"].notna()
        ]

        row["future_crossing_sample_count"] = len(valid_future)

        row["mean_warning_time_sec"] = (
            valid_future["warning_time_sec"].mean()
            if len(valid_future) else np.nan
        )

        row["median_warning_time_sec"] = (
            valid_future["warning_time_sec"].median()
            if len(valid_future) else np.nan
        )

        # This is a scenario pass rate across all episode rows,
        # not a validated probability of real-world success.
        row["dispatch_in_time_percent_all_scenarios"] = (
            100 * row["dispatch_in_time_count"] / total
            if total else np.nan
        )

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    summary.to_csv(SUMMARY_OUTPUT, index=False)

    # --------------------------------------------------------
    # STEP 4: PRINT SUMMARY
    # --------------------------------------------------------

    summary = pd.read_csv(SUMMARY_OUTPUT)

    display_columns = [
        "hypothetical_threshold",
        "relay_distance_m",
        "relay_speed_mps",
        "setup_time_sec",
        "reconnection_time_sec",
        "safety_margin_sec",
        "dispatch_in_time_count",
        "dispatch_too_late_count",
        "threshold_already_breached_count",
        "no_crossing_observed_count",
        "dispatch_in_time_percent_all_scenarios",
    ]

    print("\nRELAY DISPATCH SCENARIO SUMMARY")
    print("=" * 110)

    print(
        summary[display_columns]
        .sort_values(
            [
                "hypothetical_threshold",
                "relay_distance_m",
                "relay_speed_mps",
                "setup_time_sec",
                "reconnection_time_sec",
                "safety_margin_sec",
            ]
        )
        .to_string(
            index=False,
            formatters={
                "hypothetical_threshold": "{:.0f}".format,
                "relay_distance_m": "{:.0f}".format,
                "relay_speed_mps": "{:.2f}".format,
                "setup_time_sec": "{:.0f}".format,
                "reconnection_time_sec": "{:.0f}".format,
                "safety_margin_sec": "{:.0f}".format,
                "dispatch_in_time_count": "{:.0f}".format,
                "dispatch_too_late_count": "{:.0f}".format,
                "threshold_already_breached_count": "{:.0f}".format,
                "no_crossing_observed_count": "{:.0f}".format,
                "dispatch_in_time_percent_all_scenarios": "{:.2f}%".format,
            },
        )
    )

    print(f"\nDetailed scenarios saved to: {EVENT_OUTPUT}")
    print(f"Summary saved to: {SUMMARY_OUTPUT}")

    print("\n" + "=" * 75)
    print("IMPORTANT LIMITATIONS")
    print("=" * 75)
    print(
        "1. Thresholds, distances, speeds, setup times, and "
        "reconnection times are hypothetical scenarios."
    )
    print(
        "2. The signal metric is the median of five channels; "
        "its relationship to actual link failure is unverified."
    )
    print(
        "3. Detection time is when the retrospective decline rule "
        "first becomes true, not when deterioration originally began."
    )
    print(
        "4. No threshold crossing in a recording does not prove "
        "that communication remained safe."
    )
    print(
        "5. These results are scenario calculations, not validated "
        "dispatch accuracy or real-world reliability."
    )


if __name__ == "__main__":
    main()
