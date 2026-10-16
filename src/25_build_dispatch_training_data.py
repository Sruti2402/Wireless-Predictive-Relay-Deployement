
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "script24" / "simulation_inputs.csv"
OUT = ROOT / "data" / "processed" / "script25"
OUT.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [20, 30, 40]
RELAY_SPEEDS_MPS = [0.2, 0.4, 0.8]
SETUP_TIMES_SEC = [1.0, 2.0, 4.0]
RECONNECT_TIMES_SEC = [1.0, 2.0]
DISPATCH_DISTANCE_M = 5.0
SAFETY_MARGIN_SEC = 2.0

# Keep examples reasonably close to the sampling interval.
MAX_DEPLOYMENT_WINDOW_SEC = (
    DISPATCH_DISTANCE_M / min(RELAY_SPEEDS_MPS)
    + max(SETUP_TIMES_SEC)
    + max(RECONNECT_TIMES_SEC)
    + SAFETY_MARGIN_SEC
)


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT}")

    df = pd.read_csv(INPUT)
    df = df.sort_values(
        ["recording_id", "elapsed_sec"]
    ).reset_index(drop=True)

    required = [
        "recording_id", "elapsed_sec",
        "signal_quality", "signal_quality_smooth",
        "signal_5s_change", "signal_5s_decline",
        "speed_mps", "distance_from_base",
        "UL_level", "UR_level", "LL_level",
        "LR_level", "C_level",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    numeric_cols = [c for c in required if c != "recording_id"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=[
            "recording_id", "elapsed_sec",
            "signal_quality", "signal_quality_smooth",
        ]
    ).copy()

    # No model input may depend on future signal samples.
    feature_cols = [
        "signal_quality",
        "signal_quality_smooth",
        "signal_5s_change",
        "signal_5s_decline",
        "speed_mps",
        "distance_from_base",
        "UL_level", "UR_level", "LL_level",
        "LR_level", "C_level",
    ]

    for col in feature_cols:
        df[col] = df[col].replace(
            [np.inf, -np.inf], np.nan
        )

    df["speed_mps"] = df["speed_mps"].fillna(0.0)
    df["signal_5s_change"] = df["signal_5s_change"].fillna(0.0)
    df["signal_5s_decline"] = df["signal_5s_decline"].fillna(0.0)

    # Build labels separately for each recording, so no future event
    # from one recording can label another recording.
    output_frames = []

    for recording_id, group in df.groupby("recording_id", sort=False):
        group = group.sort_values("elapsed_sec").reset_index(drop=True)

        times = group["elapsed_sec"].to_numpy(dtype=float)
        signal = group["signal_quality_smooth"].to_numpy(dtype=float)

        for threshold in THRESHOLDS:
            low_indices = np.flatnonzero(signal <= threshold)
            low_times = times[low_indices]

            # Generate a scenario for every deployment configuration.
            for relay_speed in RELAY_SPEEDS_MPS:
                for setup_sec in SETUP_TIMES_SEC:
                    for reconnect_sec in RECONNECT_TIMES_SEC:
                        travel_sec = DISPATCH_DISTANCE_M / relay_speed
                        deployment_sec = travel_sec + setup_sec + reconnect_sec
                        decision_window_sec = (
                            deployment_sec + SAFETY_MARGIN_SEC
                        )

                        # Find the first future low-signal sample for
                        # every row. side="right" excludes the current
                        # timestamp, making this a future-event label.
                        next_low_pos = np.searchsorted(
                            low_times, times, side="right"
                        )

                        has_future_crossing = (
                            next_low_pos < len(low_times)
                        )

                        time_to_crossing = np.full(len(group), np.nan)
                        valid = np.flatnonzero(has_future_crossing)

                        time_to_crossing[valid] = (
                            low_times[next_low_pos[valid]] - times[valid]
                        )

                        currently_above_threshold = signal > threshold

                        # Positive means dispatch now while enough time
                        # remains for deployment, but within the safety window.
                        dispatch_now = (
                            currently_above_threshold
                            & has_future_crossing
                            & (time_to_crossing >= deployment_sec)
                            & (time_to_crossing <= decision_window_sec)
                        ).astype(int)

                        scenario = group[feature_cols].copy()
                        scenario.insert(0, "recording_id", recording_id)
                        scenario.insert(1, "elapsed_sec", times)

                        scenario["signal_threshold"] = threshold
                        scenario["relay_speed_mps"] = relay_speed
                        scenario["setup_time_sec"] = setup_sec
                        scenario["reconnect_time_sec"] = reconnect_sec
                        scenario["dispatch_distance_m"] = DISPATCH_DISTANCE_M
                        scenario["travel_time_sec"] = travel_sec
                        scenario["deployment_time_sec"] = deployment_sec
                        scenario["safety_margin_sec"] = SAFETY_MARGIN_SEC

                        scenario["time_to_next_low_signal_sec"] = (
                            time_to_crossing
                        )
                        scenario["dispatch_now"] = dispatch_now

                        # Already-low samples and the absence of any
                        # future crossing are kept as negative examples.
                        # A later version can explicitly model recovery,
                        # persistent outage and relay-chain placement.
                        output_frames.append(scenario)

    if not output_frames:
        raise RuntimeError("No training scenarios were generated.")

    training = pd.concat(output_frames, ignore_index=True)

    output_path = OUT / "dispatch_training_data.csv"
    training.to_csv(output_path, index=False)

    # Summary by recording and threshold helps expose class imbalance.
    summary = (
        training.groupby(
            ["recording_id", "signal_threshold"], as_index=False
        )
        .agg(
            examples=("dispatch_now", "size"),
            dispatch_positive=("dispatch_now", "sum"),
            positive_rate=("dispatch_now", "mean"),
        )
    )

    summary.to_csv(OUT / "dispatch_label_summary.csv", index=False)

    print("=" * 72)
    print("SCRIPT 25 — DISPATCH TRAINING DATA")
    print("=" * 72)
    print(f"Training examples: {len(training):,}")
    print(f"Recordings: {training['recording_id'].nunique()}")
    print(f"Scenario configurations: "
          f"{training[['signal_threshold', 'relay_speed_mps', 'setup_time_sec', 'reconnect_time_sec']].drop_duplicates().shape[0]}")
    print(f"Positive dispatch labels: {int(training['dispatch_now'].sum()):,}")
    print(f"Positive label rate: {training['dispatch_now'].mean():.2%}")

    print("\nLABEL COUNTS")
    print(training["dispatch_now"].value_counts().sort_index().to_string())

    print("\nSUMMARY BY RECORDING AND THRESHOLD")
    print(summary.round(3).to_string(index=False))

    print("\nSaved:")
    print(output_path)
    print(OUT / "dispatch_label_summary.csv")

    print("\nIMPORTANT:")
    print("- Labels use future observed signal crossings.")
    print("- Model features use current/past measurements only.")
    print("- These are proxy dispatch labels, not real relay outcomes.")
    print("- Evaluate by holding out entire recordings.")


if __name__ == "__main__":
    main()
