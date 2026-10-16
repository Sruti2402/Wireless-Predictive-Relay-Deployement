from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "combined_repaired.csv"
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SIGNALS = [
    "UL_level", "UR_level", "LL_level", "LR_level", "C_level",
    "UL_level_a", "UR_level_a", "LL_level_a", "LR_level_a", "C_level_a",
]


def main():
    df = pd.read_csv(INPUT)

    required = [
        "recording_id", "timestamp", "Feedback",
        "robot_pos_x", "robot_pos_y", *SIGNALS,
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.sort_values(
        ["recording_id", "timestamp", "temp_step"],
        kind="stable"
    ).reset_index(drop=True)

    df["Feedback"] = pd.to_numeric(df["Feedback"], errors="coerce")

    print("=" * 65)
    print("FEEDBACK EVENT ANALYSIS")
    print("=" * 65)
    print("Rows:", len(df))
    print("Feedback = 1:", int((df["Feedback"] == 1).sum()))
    print("Feedback = 0:", int((df["Feedback"] == 0).sum()))

    print("\nEvents by recording:")
    print(
        df.groupby("recording_id")["Feedback"]
        .agg(
            rows="size",
            feedback_events=lambda x: int((x == 1).sum())
        )
        .to_string()
    )

    # Compare measurements in positive and negative feedback samples.
    print("\nSignal measurements grouped by Feedback:")
    print(
        df.groupby("Feedback")[SIGNALS]
        .mean()
        .round(2)
        .to_string()
    )

    # Save every positive-feedback sample with up to five preceding
    # and five following rows from the same recording.
    event_parts = []

    for recording_id, group in df.groupby("recording_id", sort=True):
        group = group.reset_index(drop=True)
        event_indices = group.index[group["Feedback"] == 1].tolist()

        for event_number, idx in enumerate(event_indices, start=1):
            start = max(0, idx - 5)
            end = min(len(group), idx + 6)

            context = group.iloc[start:end].copy()
            context["event_recording"] = recording_id
            context["event_number"] = event_number
            context["event_row_index"] = idx
            context["rows_from_event"] = context.index - idx
            event_parts.append(context)

    if event_parts:
        event_context = pd.concat(event_parts, ignore_index=True)
        event_path = OUTPUT_DIR / "feedback_event_context.csv"
        event_context.to_csv(event_path, index=False)
        print("\nSaved event context:", event_path)
        print("Context rows:", len(event_context))
    else:
        print("\nNo Feedback = 1 events found.")

    # Summarize signal trends before and after each event.
    trend_rows = []

    for recording_id, group in df.groupby("recording_id", sort=True):
        group = group.reset_index(drop=True)
        event_indices = group.index[group["Feedback"] == 1].tolist()

        for idx in event_indices:
            before = group.iloc[max(0, idx - 10):idx]
            event = group.iloc[[idx]]
            after = group.iloc[idx + 1:min(len(group), idx + 11)]

            row = {
                "recording_id": recording_id,
                "event_index": idx,
                "event_timestamp": group.loc[idx, "timestamp"],
                "event_x": group.loc[idx, "robot_pos_x"],
                "event_y": group.loc[idx, "robot_pos_y"],
                "before_samples": len(before),
                "after_samples": len(after),
            }

            for signal in SIGNALS:
                row[f"{signal}_before_mean"] = (
                    before[signal].mean() if len(before) else float("nan")
                )
                row[f"{signal}_at_event"] = event[signal].iloc[0]
                row[f"{signal}_after_mean"] = (
                    after[signal].mean() if len(after) else float("nan")
                )

            trend_rows.append(row)

    trends = pd.DataFrame(trend_rows)
    trends_path = OUTPUT_DIR / "feedback_event_trends.csv"
    trends.to_csv(trends_path, index=False)

    print("Saved event trends:", trends_path)

    if not trends.empty:
        print("\nFirst 15 event summaries:")
        display_cols = [
            "recording_id", "event_timestamp", "event_x", "event_y",
            "UL_level_before_mean", "UL_level_at_event",
            "UL_level_after_mean",
            "UL_level_a_before_mean", "UL_level_a_at_event",
            "UL_level_a_after_mean",
        ]
        print(trends[display_cols].head(15).round(3).to_string(index=False))

    print("\nAnalysis complete.")
    print("These events are not confirmed link failures until the")
    print("meaning of the Feedback field is verified.")


if __name__ == "__main__":
    main()