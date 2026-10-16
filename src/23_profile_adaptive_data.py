
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "combined_repaired.csv"
OUT = ROOT / "data" / "processed" / "script23"
OUT.mkdir(parents=True, exist_ok=True)

SIGNAL_COLS = [
    "UL_level", "UR_level", "LL_level", "LR_level", "C_level"
]

POSITION_COLS = ["robot_pos_x", "robot_pos_y"]


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing input: {INPUT}")

    df = pd.read_csv(INPUT)
    df.columns = df.columns.str.strip()

    required = {
        "recording_id", "elapsed_sec",
        *POSITION_COLS, *SIGNAL_COLS
    }
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    for col in [
        "elapsed_sec", *POSITION_COLS, *SIGNAL_COLS
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=["recording_id", "elapsed_sec", *POSITION_COLS]
    ).copy()

    df = df.sort_values(
        ["recording_id", "elapsed_sec"]
    ).reset_index(drop=True)

    # Calculate movement independently within each recording.
    grouped = df.groupby("recording_id", sort=False)

    df["dt_sec"] = grouped["elapsed_sec"].diff()
    df["dx_m"] = grouped["robot_pos_x"].diff()
    df["dy_m"] = grouped["robot_pos_y"].diff()

    # Ignore duplicate or reversed timestamps for speed calculations.
    valid_dt = df["dt_sec"] > 0
    df["step_distance_m"] = np.hypot(df["dx_m"], df["dy_m"])
    df["speed_mps"] = np.nan
    df.loc[valid_dt, "speed_mps"] = (
        df.loc[valid_dt, "step_distance_m"]
        / df.loc[valid_dt, "dt_sec"]
    )

    summaries = []

    for recording_id, g in df.groupby("recording_id"):
        g = g.sort_values("elapsed_sec")

        x = g["robot_pos_x"].dropna()
        y = g["robot_pos_y"].dropna()
        speeds = g["speed_mps"].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()

        path_length = g["step_distance_m"].sum(min_count=1)

        displacement = (
            np.hypot(x.iloc[-1] - x.iloc[0],
                     y.iloc[-1] - y.iloc[0])
            if len(x) >= 2 and len(y) >= 2 else np.nan
        )

        row = {
            "recording_id": recording_id,
            "rows": len(g),
            "duration_sec": g["elapsed_sec"].max()
                            - g["elapsed_sec"].min(),
            "x_min": x.min(),
            "x_max": x.max(),
            "y_min": y.min(),
            "y_max": y.max(),
            "x_range": x.max() - x.min(),
            "y_range": y.max() - y.min(),
            "unique_position_pairs": len(
                g[POSITION_COLS].drop_duplicates()
            ),
            "path_length_m": path_length,
            "net_displacement_m": displacement,
            "median_speed_mps": speeds.median(),
            "p95_speed_mps": speeds.quantile(0.95),
            "positive_dt_fraction": (g["dt_sec"] > 0).mean(),
        }

        for col in SIGNAL_COLS:
            values = g[col].dropna()
            row[f"{col}_count"] = len(values)
            row[f"{col}_median"] = values.median()
            row[f"{col}_std"] = values.std()
            row[f"{col}_min"] = values.min()
            row[f"{col}_max"] = values.max()

        summaries.append(row)

    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "recording_profile.csv", index=False)

    # Basic quality checks for each signal channel.
    quality = []
    for recording_id, g in df.groupby("recording_id"):
        for col in SIGNAL_COLS:
            quality.append({
                "recording_id": recording_id,
                "signal": col,
                "missing_count": int(g[col].isna().sum()),
                "unique_values": int(g[col].nunique()),
                "constant_signal": g[col].nunique(dropna=True) <= 1,
            })

    pd.DataFrame(quality).to_csv(
        OUT / "signal_quality.csv", index=False
    )

    df.to_csv(OUT / "movement_features.csv", index=False)

    print("=" * 72)
    print("SCRIPT 23 — ADAPTIVE DATA PROFILE")
    print("=" * 72)
    print(f"Rows inspected: {len(df):,}")
    print(f"Recordings: {df['recording_id'].nunique()}")
    print("\nMOVEMENT AND SIGNAL SUMMARY")
    print(summary.round(4).to_string(index=False))

    print("\nTIMESTAMP CHECKS")
    print("Duplicate timestamps:",
          int(df.duplicated(["recording_id", "elapsed_sec"]).sum()))
    print("Non-positive time differences:",
          int((df["dt_sec"].dropna() <= 0).sum()))

    print("\nSaved to:", OUT)
    print("- recording_profile.csv")
    print("- signal_quality.csv")
    print("- movement_features.csv")


if __name__ == "__main__":
    main()

