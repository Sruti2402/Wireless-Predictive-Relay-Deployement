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

WINDOW_SECONDS = 5.0


def main():
    df = pd.read_csv(INPUT)

    required = {
        "recording_id",
        "timestamp",
        "elapsed_sec",
        "Feedback",
        "robot_pos_x",
        "robot_pos_y",
        *SIGNALS,
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["Feedback"] = pd.to_numeric(df["Feedback"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["elapsed_sec"] = pd.to_numeric(df["elapsed_sec"], errors="coerce")

    df = df.dropna(
        subset=["recording_id", "timestamp", "elapsed_sec", "Feedback"]
    )

    df = df.sort_values(
        ["recording_id", "timestamp"]
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # 1. Identify contiguous runs of Feedback = 1
    # ---------------------------------------------------------
    run_records = []

    for recording_id, group in df.groupby("recording_id", sort=False):
        group = group.sort_values("timestamp").copy()
        positive = group["Feedback"].eq(1)

        # A new run starts when a positive row follows a non-positive row.
        starts = positive & ~positive.shift(1, fill_value=False)
        run_ids = starts.cumsum()

        for run_id in run_ids[positive].unique():
            run = group.loc[positive & run_ids.eq(run_id)]

            run_records.append({
                "recording_id": recording_id,
                "run_start_timestamp": run["timestamp"].iloc[0],
                "run_end_timestamp": run["timestamp"].iloc[-1],
                "run_start_elapsed_sec": run["elapsed_sec"].iloc[0],
                "run_end_elapsed_sec": run["elapsed_sec"].iloc[-1],
                "positive_rows": len(run),
                "duration_sec": (
                    run["elapsed_sec"].max()
                    - run["elapsed_sec"].min()
                ),
                "x": run["robot_pos_x"].mean(),
                "y": run["robot_pos_y"].mean(),
            })

    runs = pd.DataFrame(run_records)
    runs_path = OUTPUT / "feedback_runs.csv"
    runs.to_csv(runs_path, index=False)

    print("=" * 65)
    print("FEEDBACK RUN ANALYSIS")
    print("=" * 65)
    print(f"Total positive rows: {(df['Feedback'] == 1).sum()}")
    print(f"Contiguous positive runs: {len(runs)}")

    if not runs.empty:
        print("\nPositive rows per run:")
        print(runs["positive_rows"].value_counts().sort_index())

        print("\nRun duration statistics (seconds):")
        print(runs["duration_sec"].describe().round(3))

    # ---------------------------------------------------------
    # 2. Compare signal levels before, at, and after each event
    # ---------------------------------------------------------
    event_records = []

    for recording_id, group in df.groupby("recording_id", sort=False):
        group = group.sort_values("timestamp").reset_index(drop=True)
        event_indices = group.index[group["Feedback"].eq(1)].tolist()

        for idx in event_indices:
            event = group.iloc[idx]
            t = event["elapsed_sec"]

            before = group[
                (group["elapsed_sec"] < t)
                & (group["elapsed_sec"] >= t - WINDOW_SECONDS)
            ]

            after = group[
                (group["elapsed_sec"] > t)
                & (group["elapsed_sec"] <= t + WINDOW_SECONDS)
            ]

            record = {
                "recording_id": recording_id,
                "event_timestamp": event["timestamp"],
                "event_elapsed_sec": t,
                "event_x": event["robot_pos_x"],
                "event_y": event["robot_pos_y"],
                "before_rows": len(before),
                "after_rows": len(after),
            }

            for signal in SIGNALS:
                before_mean = before[signal].mean() if not before.empty else np.nan
                at_event = event[signal]
                after_mean = after[signal].mean() if not after.empty else np.nan

                record[f"{signal}_before_5s"] = before_mean
                record[f"{signal}_at_event"] = at_event
                record[f"{signal}_after_5s"] = after_mean

                record[f"{signal}_change_before_to_event"] = (
                    at_event - before_mean
                    if pd.notna(before_mean) and pd.notna(at_event)
                    else np.nan
                )

            event_records.append(record)

    events = pd.DataFrame(event_records)
    events_path = OUTPUT / "feedback_event_5s_analysis.csv"
    events.to_csv(events_path, index=False)

    print("\n" + "=" * 65)
    print("SIGNAL CHANGE IN THE 5 SECONDS BEFORE FEEDBACK = 1")
    print("=" * 65)

    for signal in SIGNALS:
        change_col = f"{signal}_change_before_to_event"
        valid = events[change_col].dropna()

        if valid.empty:
            continue

        print(f"\n{signal}")
        print(f"  Events with usable preceding window: {len(valid)}")
        print(f"  Mean change:   {valid.mean():.3f}")
        print(f"  Median change: {valid.median():.3f}")
        print(f"  Signal declined: {(valid < 0).mean() * 100:.1f}%")
        print(f"  Signal increased: {(valid > 0).mean() * 100:.1f}%")
        print(f"  No change: {(valid == 0).mean() * 100:.1f}%")

    print("\nSaved files:")
    print(runs_path)
    print(events_path)
    print("\nAnalysis complete.")
    print(
        "Reminder: this analysis does not establish what Feedback = 1 means."
    )


if __name__ == "__main__":
    main()
