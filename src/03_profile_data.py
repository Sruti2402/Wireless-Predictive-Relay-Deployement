
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(RAW_DIR.glob("*.datalog"))
if not files:
    raise FileNotFoundError(f"No .datalog files found in {RAW_DIR}")

frames = []

for path in files:
    df = pd.read_csv(
        path,
        sep=r"\s+",
        engine="python",
        comment="#"
    )
    df["recording_id"] = path.stem
    frames.append(df)

data = pd.concat(frames, ignore_index=True)

print("TOTAL ROWS:", len(data))
print("TOTAL COLUMNS:", len(data.columns))
print("\nCOLUMNS:")
print(data.columns.tolist())

# Save combined data without altering the source recordings.
combined_path = OUT_DIR / "combined_raw.csv"
data.to_csv(combined_path, index=False)
print("\nSaved:", combined_path)

# Profile important fields.
position_cols = ["robot_pos_x", "robot_pos_y"]
signal_cols = [
    "UL_level", "UR_level", "LL_level",
    "LR_level", "C_level",
    "UL_level_a", "UR_level_a", "LL_level_a",
    "LR_level_a", "C_level_a"
]
time_cols = ["temp_step", "temp_sec", "temp_nsec"]

for col in position_cols + signal_cols + time_cols:
    data[col] = pd.to_numeric(data[col], errors="coerce")

print("\n--- MISSING VALUES ---")
print(data[position_cols + signal_cols + time_cols].isna().sum())

print("\n--- POSITION STATISTICS ---")
print(data.groupby("recording_id")[position_cols].agg(
    ["min", "max", "nunique", "std"]
).to_string())

print("\n--- SIGNAL STATISTICS ---")
print(data[signal_cols].describe().round(3).to_string())

print("\n--- TIMESTAMP STATISTICS ---")
print(data.groupby("recording_id")[time_cols].agg(
    ["min", "max", "nunique", "std"]
).to_string())

print("\n--- FIRST 5 RAW ROWS ---")
print(data.head(5).to_string(index=False))

print("\n--- FIRST 5 ROWS OF DATASET1 ---")
print(
    data[data["recording_id"] == "Dataset1"]
    .head(5)
    .to_string(index=False)
)

# Test whether temp_step is a usable increasing timestamp.
print("\n--- TEMP_STEP ORDER CHECK ---")
for name, group in data.groupby("recording_id", sort=False):
    steps = group["temp_step"]
    differences = steps.diff().dropna()
    print(
        f"{name}: increasing fraction="
        f"{(differences > 0).mean():.3f}, "
        f"unique={steps.nunique()}/{len(steps)}"
    )
