from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "combined_repaired.csv"
OUTPUT = ROOT / "data" / "processed" / "forecast_dataset.csv"

SIGNALS = [
    "UL_level",
    "UR_level",
    "LL_level",
    "LR_level",
    "C_level",
    "UL_level_a",
    "UR_level_a",
    "LL_level_a",
    "LR_level_a",
    "C_level_a",
]

HORIZONS_SECONDS = [1, 3, 5]
LAGS_SECONDS = [1, 3, 5]
MAX_TIME_ERROR_SECONDS = 0.5


def nearest_past_index(times, target_time):
    """Return the most recent index at or before target_time."""
    idx = np.searchsorted(times, target_time, side="right") - 1

    if idx < 0:
        return None

    if target_time - times[idx] > MAX_TIME_ERROR_SECONDS:
        return None

    return idx


def main():
    df = pd.read_csv(INPUT)

    required = [
        "recording_id",
        "elapsed_sec",
        "robot_pos_x",
        "robot_pos_y",
        *SIGNALS,
    ]

    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    for col in required[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=["recording_id", "elapsed_sec", "robot_pos_x", "robot_pos_y"]
    )

    output_rows = []

    for recording_id, group in df.groupby("recording_id", sort=True):
        group = group.sort_values("elapsed_sec")

        # Remove duplicate timestamps within a recording.
        group = group.drop_duplicates(
            subset=["elapsed_sec"], keep="last"
        ).reset_index(drop=True)

        times = group["elapsed_sec"].to_numpy(dtype=float)
        if len(times) < 10:
            continue

        signals = group[SIGNALS].to_numpy(dtype=float)
        positions = group[
            ["robot_pos_x", "robot_pos_y"]
        ].to_numpy(dtype=float)

        for i, t in enumerate(times):
            row = {
                "recording_id": recording_id,
                "elapsed_sec": t,
                "robot_pos_x": positions[i, 0],
                "robot_pos_y": positions[i, 1],
            }

            # Current measurements available at prediction time.
            for j, signal in enumerate(SIGNALS):
                row[f"current_{signal}"] = signals[i, j]

            # Historical measurements and movement features.
            history_valid = True

            for lag in LAGS_SECONDS:
                past_idx = nearest_past_index(times, t - lag)

                if past_idx is None:
                    history_valid = False
                    break

                for j, signal in enumerate(SIGNALS):
                    row[f"{signal}_lag_{lag}s"] = signals[past_idx, j]

                dx = positions[i, 0] - positions[past_idx, 0]
                dy = positions[i, 1] - positions[past_idx, 1]

                row[f"displacement_{lag}s"] = np.hypot(dx, dy)
                row[f"velocity_x_{lag}s"] = dx / lag
                row[f"velocity_y_{lag}s"] = dy / lag

            if not history_valid:
                continue

            # Future measurements are targets, never input features.
            future_valid = True

            for horizon in HORIZONS_SECONDS:
                future_idx = np.searchsorted(
                    times, t + horizon, side="left"
                )

                if future_idx >= len(times):
                    future_valid = False
                    break

                time_error = times[future_idx] - (t + horizon)

                if time_error < 0 or time_error > MAX_TIME_ERROR_SECONDS:
                    future_valid = False
                    break

                for j, signal in enumerate(SIGNALS):
                    row[f"future_{signal}_{horizon}s"] = (
                        signals[future_idx, j]
                    )

            if future_valid:
                output_rows.append(row)

    result = pd.DataFrame(output_rows)

    if result.empty:
        raise RuntimeError(
            "No forecast rows were created. Check elapsed_sec values "
            "and the sampling interval."
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)

    print("=" * 65)
    print("FORECAST DATASET")
    print("=" * 65)
    print(f"Rows: {len(result)}")
    print(f"Columns: {len(result.columns)}")
    print(f"Recordings: {result['recording_id'].nunique()}")

    print("\nRows by recording:")
    print(result["recording_id"].value_counts().sort_index())

    print("\nMissing values in generated dataset:")
    print(int(result.isna().sum().sum()))

    print("\nTarget columns:")
    for horizon in HORIZONS_SECONDS:
        print(f"\n{horizon}-second horizon:")
        print([
            f"future_{signal}_{horizon}s"
            for signal in SIGNALS
        ])

    print(f"\nSaved: {OUTPUT}")
    print("\nDataset creation complete.")


if __name__ == "__main__":
    main()
