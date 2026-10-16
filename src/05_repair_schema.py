from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Each data row contains 23 values, although the original
# header contains only 21 column names.
COLUMNS = [
    "temp_step",
    "timestamp_sec",
    "timestamp_nsec",
    "robot_pos_x",
    "robot_pos_y",
    "robot_w_x",
    "robot_w_y",
    "robot_w_z",
    "robot_w_w",
    "theta_p",
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
    "extra_1",
    "extra_2",
    "Feedback",
]


def read_recording(path):
    rows = []
    malformed = []

    with path.open(
        "r",
        encoding="utf-8",
        errors="replace"
    ) as file:
        for line_number, line in enumerate(file, start=1):
            fields = line.split()

            if not fields:
                continue

            # Ignore the original header and comment lines.
            if fields[0].startswith("#"):
                continue

            if fields[0] == "temp_step":
                continue

            if len(fields) != len(COLUMNS):
                malformed.append(
                    (line_number, len(fields))
                )
                continue

            try:
                rows.append([float(value) for value in fields])
            except ValueError:
                malformed.append(
                    (line_number, len(fields))
                )

    df = pd.DataFrame(rows, columns=COLUMNS)
    df["recording_id"] = path.stem

    print(
        f"{path.name}: {len(df)} valid rows, "
        f"{len(malformed)} malformed rows"
    )

    if malformed:
        print("First malformed rows:", malformed[:5])

    return df


def main():
    files = sorted(RAW_DIR.glob("*.datalog"))

    if not files:
        raise FileNotFoundError(
            f"No .datalog files found in {RAW_DIR}"
        )

    frames = [read_recording(path) for path in files]
    data = pd.concat(frames, ignore_index=True)

    # Construct a timestamp in seconds. Keep each recording
    # separate because timestamps may reset between recordings.
    data["timestamp"] = (
        data["timestamp_sec"]
        + data["timestamp_nsec"] / 1_000_000_000
    )

    data["elapsed_sec"] = (
        data["timestamp"]
        - data.groupby("recording_id")["timestamp"].transform("first")
    )

    output_path = OUT_DIR / "combined_repaired.csv"
    data.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print("REPAIR SUMMARY")
    print("=" * 60)

    print("Total recordings:", len(files))
    print("Total rows:", len(data))
    print("Total columns:", len(data.columns))
    print("Saved to:", output_path)

    print("\nPosition statistics by recording:")
    print(
        data.groupby("recording_id")[
            ["robot_pos_x", "robot_pos_y"]
        ].agg(["min", "max", "nunique"]).to_string()
    )

    print("\nTimestamp statistics by recording:")
    print(
        data.groupby("recording_id")[
            ["timestamp", "elapsed_sec"]
        ].agg(["min", "max", "nunique"]).to_string()
    )

    print("\nSignal statistics:")
    signal_columns = [
        "UL_level", "UR_level", "LL_level",
        "LR_level", "C_level",
        "UL_level_a", "UR_level_a", "LL_level_a",
        "LR_level_a", "C_level_a"
    ]
    print(data[signal_columns].describe().round(3).to_string())

    print("\nFirst five repaired rows:")
    print(data.head().to_string(index=False))


if __name__ == "__main__":
    main()