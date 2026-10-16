from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "combined_repaired.csv"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find repaired dataset: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("=" * 70)
    print("DATASET SCHEMA")
    print("=" * 70)

    print("Shape:", df.shape)
    print("\nExact column names:")
    for index, column in enumerate(df.columns, start=1):
        print(f"{index:02d}. {column!r}")

    required = [
        "recording_id",
        "timestamp",
        "elapsed_sec",
        "Feedback",
        "extra_1",
        "extra_2",
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    print("\n" + "=" * 70)
    print("UNKNOWN FIELDS AND FEEDBACK")
    print("=" * 70)

    for col in ["extra_1", "extra_2", "Feedback"]:
        print(f"\n{col}:")
        print("Unique values:", sorted(df[col].dropna().unique().tolist())[:50])
        print("\nValue counts:")
        print(df[col].value_counts(dropna=False).head(20).to_string())

    print("\n" + "=" * 70)
    print("FEEDBACK VALUES BY RECORDING")
    print("=" * 70)

    for recording, group in df.groupby("recording_id", sort=True):
        print(f"\n{recording}")
        print(group["Feedback"].value_counts(dropna=False).to_string())

    print("\n" + "=" * 70)
    print("SIGNAL SUMMARY BY RECORDING")
    print("=" * 70)

    signal_cols = [
        "UL_level", "UR_level", "LL_level",
        "LR_level", "C_level",
        "UL_level_a", "UR_level_a", "LL_level_a",
        "LR_level_a", "C_level_a",
    ]

    summary = df.groupby("recording_id")[signal_cols].agg(
        ["min", "median", "max", "nunique"]
    )

    print(summary.round(2).to_string())

    print("\n" + "=" * 70)
    print("TIMESTAMP ORDER AND SAMPLING INTERVALS")
    print("=" * 70)

    for recording, group in df.groupby("recording_id", sort=True):
        group = group.sort_values("timestamp").copy()
        dt = group["timestamp"].diff().dropna()

        print(f"\n{recording}")
        print("Rows:", len(group))
        print("Duration (s):", round(group["timestamp"].max() - group["timestamp"].min(), 3))
        print("Duplicate timestamps:", int(group["timestamp"].duplicated().sum()))

        if not dt.empty:
            print("Non-positive intervals:", int((dt <= 0).sum()))
            print("Median interval (s):", round(dt.median(), 4))
            print("95th percentile interval (s):", round(dt.quantile(0.95), 4))
            print("Maximum interval (s):", round(dt.max(), 4))

    print("\n" + "=" * 70)
    print("FIRST AND LAST ROW OF EACH RECORDING")
    print("=" * 70)

    preview_cols = [
        "recording_id",
        "temp_step",
        "timestamp",
        "elapsed_sec",
        "robot_pos_x",
        "robot_pos_y",
        "UL_level",
        "C_level",
        "UL_level_a",
        "C_level_a",
        "extra_1",
        "extra_2",
        "Feedback",
    ]

    for recording, group in df.groupby("recording_id", sort=True):
        group = group.sort_values("timestamp")
        print(f"\n{recording} — first row")
        print(group[preview_cols].head(1).to_string(index=False))
        print(f"{recording} — last row")
        print(group[preview_cols].tail(1).to_string(index=False))


if __name__ == "__main__":
    main()