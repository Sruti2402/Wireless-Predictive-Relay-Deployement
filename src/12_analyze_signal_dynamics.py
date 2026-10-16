from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "combined_repaired.csv"
OUTPUT = ROOT / "data" / "processed"

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


def main():
    df = pd.read_csv(INPUT)

    for col in [
        "elapsed_sec",
        "robot_pos_x",
        "robot_pos_y",
        *SIGNALS,
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=["recording_id", "elapsed_sec", "robot_pos_x", "robot_pos_y"]
    )

    df = df.sort_values(
        ["recording_id", "elapsed_sec"]
    ).drop_duplicates(
        subset=["recording_id", "elapsed_sec"],
        keep="last",
    )

    recording_summaries = []
    signal_summaries = []

    for recording_id, group in df.groupby("recording_id", sort=True):
        group = group.sort_values("elapsed_sec").copy()

        dt = group["elapsed_sec"].diff()
        dx = group["robot_pos_x"].diff()
        dy = group["robot_pos_y"].diff()

        # Calculate approximate planar speed only for positive time steps.
        valid_dt = dt.where(dt > 0)
        group["robot_speed"] = np.hypot(dx, dy) / valid_dt

        recording_summaries.append({
            "recording_id": recording_id,
            "rows": len(group),
            "duration_sec": (
                group["elapsed_sec"].max()
                - group["elapsed_sec"].min()
            ),
            "median_sample_interval_sec": dt[dt > 0].median(),
            "median_robot_speed": group["robot_speed"].median(),
            "p95_robot_speed": group["robot_speed"].quantile(0.95),
            "total_planar_distance": np.hypot(dx, dy).sum(),
        })

        for signal in SIGNALS:
            values = group[signal]
            changes = values.diff().abs()

            valid = pd.DataFrame({
                "signal_change": changes,
                "robot_speed": group["robot_speed"],
            }).dropna()

            correlation = (
                valid["signal_change"].corr(valid["robot_speed"])
                if len(valid) > 2
                else np.nan
            )

            signal_summaries.append({
                "recording_id": recording_id,
                "signal": signal,
                "mean": values.mean(),
                "std": values.std(),
                "min": values.min(),
                "median": values.median(),
                "max": values.max(),
                "p01": values.quantile(0.01),
                "p99": values.quantile(0.99),
                "median_absolute_change": changes.median(),
                "p95_absolute_change": changes.quantile(0.95),
                "p99_absolute_change": changes.quantile(0.99),
                "abs_change_speed_correlation": correlation,
            })

    recording_df = pd.DataFrame(recording_summaries)
    signal_df = pd.DataFrame(signal_summaries)

    recording_path = OUTPUT / "recording_motion_summary.csv"
    signal_path = OUTPUT / "signal_dynamics_summary.csv"

    recording_df.to_csv(recording_path, index=False)
    signal_df.to_csv(signal_path, index=False)

    print("=" * 72)
    print("RECORDING MOTION SUMMARY")
    print("=" * 72)
    print(recording_df.round(3).to_string(index=False))

    print("\n" + "=" * 72)
    print("SIGNAL DYNAMICS SUMMARY")
    print("=" * 72)

    for signal in SIGNALS:
        subset = signal_df[signal_df["signal"] == signal]

        print(f"\n{signal}")
        print(
            subset[
                [
                    "recording_id",
                    "mean",
                    "std",
                    "min",
                    "median",
                    "max",
                    "p95_absolute_change",
                    "p99_absolute_change",
                    "abs_change_speed_correlation",
                ]
            ].round(3).to_string(index=False)
        )

    print("\nSaved:")
    print(recording_path)
    print(signal_path)
    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
