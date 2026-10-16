
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "combined_repaired.csv"
OUT = ROOT / "data" / "processed" / "script24"
OUT.mkdir(parents=True, exist_ok=True)

POSITION_COLS = ["robot_pos_x", "robot_pos_y"]
SIGNAL_COLS = [
    "UL_level", "UR_level", "LL_level", "LR_level", "C_level"
]

# Preliminary simulation settings, not validated real-world limits.
MIN_POSITION_RANGE = 1.0
SIGNAL_THRESHOLDS = [20, 30, 40]
TREND_LOOKBACK_SEC = 5.0


def add_past_trend(group):
    """Compute a trailing signal trend without using future observations."""
    group = group.sort_values("elapsed_sec").copy()

    times = group["elapsed_sec"].to_numpy(dtype=float)
    signal = group["signal_quality"].to_numpy(dtype=float)

    past_signal = np.full(len(group), np.nan)

    for i, t in enumerate(times):
        target_time = t - TREND_LOOKBACK_SEC
        j = np.searchsorted(times, target_time, side="right") - 1

        if j >= 0 and times[j] < t:
            past_signal[i] = signal[j]

    group["signal_5s_change"] = signal - past_signal
    group["signal_5s_decline"] = -group["signal_5s_change"]

    return group


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT}")

    df = pd.read_csv(INPUT)
    df.columns = df.columns.str.strip()

    required = {
        "recording_id", "elapsed_sec",
        *POSITION_COLS, *SIGNAL_COLS
    }
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    numeric_cols = ["elapsed_sec", *POSITION_COLS, *SIGNAL_COLS]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=["recording_id", "elapsed_sec", *POSITION_COLS]
    ).copy()

    # Stable ordering makes duplicate removal deterministic.
    df = df.sort_values(
        ["recording_id", "elapsed_sec", "temp_step"],
        kind="stable"
    )

    before = len(df)

    # Keep one sample per recording/time. No timestamps are compared
    # across different recordings.
    df = df.drop_duplicates(
        subset=["recording_id", "elapsed_sec"],
        keep="first"
    ).copy()

    duplicates_removed = before - len(df)

    # Recordings 6 and 7 barely move compared with recordings 1–5.
    # Exclude them from the initial mobile-explorer simulation inputs.
    # This is a pragmatic filter; investigate their meaning separately.
    recording_ranges = (
        df.groupby("recording_id")
        .agg(
            x_range=("robot_pos_x", lambda s: s.max() - s.min()),
            y_range=("robot_pos_y", lambda s: s.max() - s.min()),
            rows=("elapsed_sec", "size"),
        )
    )

    recording_ranges["max_position_range"] = recording_ranges[
        ["x_range", "y_range"]
    ].max(axis=1)

    eligible_ids = recording_ranges.index[
        recording_ranges["max_position_range"] >= MIN_POSITION_RANGE
    ].tolist()

    df = df[df["recording_id"].isin(eligible_ids)].copy()

    df = df.sort_values(
        ["recording_id", "elapsed_sec"]
    ).reset_index(drop=True)

    # Assumption: the five channel values are comparable signal-quality
    # scores where larger values mean stronger/better signal.
    # Verify this interpretation against the dataset documentation.
    df["signal_quality"] = df[SIGNAL_COLS].median(axis=1)

    df["signal_channels_available"] = df[SIGNAL_COLS].notna().sum(axis=1)

    # Assumption for the first simulator: base station is at (0, 0).
    # Replace these coordinates when the real base location is known.
    df["distance_from_base"] = np.hypot(
        df["robot_pos_x"], df["robot_pos_y"]
    )

    grouped = df.groupby("recording_id", sort=False)

    df["dt_sec"] = grouped["elapsed_sec"].diff()
    df["dx_m"] = grouped["robot_pos_x"].diff()
    df["dy_m"] = grouped["robot_pos_y"].diff()

    df["step_distance"] = np.hypot(df["dx_m"], df["dy_m"])
    df["speed_mps"] = np.nan

    valid_dt = df["dt_sec"] > 0
    df.loc[valid_dt, "speed_mps"] = (
        df.loc[valid_dt, "step_distance"] / df.loc[valid_dt, "dt_sec"]
    )

    # Smooth only with current and previous samples.
    df["signal_quality_smooth"] = (
        df.groupby("recording_id", sort=False)["signal_quality"]
        .transform(lambda s: s.rolling(9, min_periods=1).median())
    )

    trend_frames = []
    for recording_id, group in df.groupby("recording_id", sort=False):
        group = add_past_trend(group.copy())
        group["recording_id"] = recording_id
        trend_frames.append(group)

    df = pd.concat(trend_frames, ignore_index=True)
    if "recording_id" not in df.columns:
        raise RuntimeError("Recording ID is missing after trend calculation.")

    df = df.sort_values(
        ["recording_id", "elapsed_sec"]
    ).reset_index(drop=True)

    for threshold in SIGNAL_THRESHOLDS:
        df[f"low_signal_{threshold}"] = (
            df["signal_quality_smooth"] <= threshold
        ).astype(int)

    output = OUT / "simulation_inputs.csv"
    df.to_csv(output, index=False)

    profile = (
        df.groupby("recording_id")
        .agg(
            rows=("elapsed_sec", "size"),
            duration_sec=("elapsed_sec", lambda s: s.max() - s.min()),
            path_length=("step_distance", "sum"),
            max_distance_from_base=("distance_from_base", "max"),
            median_speed=("speed_mps", "median"),
            p95_speed=("speed_mps", lambda s: s.quantile(0.95)),
            signal_median=("signal_quality", "median"),
            signal_min=("signal_quality", "min"),
            signal_std=("signal_quality", "std"),
        )
        .reset_index()
    )

    profile.to_csv(OUT / "simulation_recording_profile.csv", index=False)

    print("=" * 72)
    print("SCRIPT 24 — SIMULATION INPUT PREPARATION")
    print("=" * 72)
    print(f"Original rows: {before:,}")
    print(f"Duplicate timestamps removed: {duplicates_removed}")
    print(f"Eligible recordings: {eligible_ids}")
    print(f"Prepared rows: {len(df):,}")

    print("\nRECORDING SUMMARY")
    print(profile.round(4).to_string(index=False))

    print("\nOUTPUTS")
    print(output)
    print(OUT / "simulation_recording_profile.csv")
    print("\nReminder: signal thresholds and base location are assumptions.")


if __name__ == "__main__":
    main()
