
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "combined_repaired.csv"
OUTPUT_DIR = ROOT / "data" / "processed" / "signal_plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

# Convert relevant fields to numeric.
numeric_cols = [
    "elapsed_sec",
    "robot_pos_x",
    "robot_pos_y",
    "Feedback",
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

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["recording_id", "elapsed_sec"])
df = df.sort_values(["recording_id", "elapsed_sec"])

main_signals = [
    "UL_level",
    "UR_level",
    "LL_level",
    "LR_level",
    "C_level",
]

a_signals = [
    "UL_level_a",
    "UR_level_a",
    "LL_level_a",
    "LR_level_a",
    "C_level_a",
]

# Report extreme values without deleting or modifying them.
print("\nEXTREME _a SIGNAL VALUES")
print("=" * 70)

for col in a_signals:
    if col not in df.columns:
        continue

    values = df[col].dropna()

    print(f"\n{col}")
    print(f"Valid readings: {len(values)}")
    print(f"Minimum: {values.min():.2f}")
    print(f"1st percentile: {values.quantile(0.01):.2f}")
    print(f"Median: {values.median():.2f}")
    print(f"99th percentile: {values.quantile(0.99):.2f}")
    print(f"Maximum: {values.max():.2f}")

    suspicious = df.loc[
        (df[col] < -100) | (df[col] > 100),
        ["recording_id", "elapsed_sec", col, "Feedback"],
    ]

    print(f"Readings outside [-100, 100]: {len(suspicious)}")

    if not suspicious.empty:
        print(suspicious.head(10).to_string(index=False))

# Plot the main signals and the _a signals separately for each recording.
for recording_id, group in df.groupby("recording_id"):
    group = group.sort_values("elapsed_sec")

    # Plot 1: main signal fields.
    fig, ax = plt.subplots(figsize=(13, 5))

    for col in main_signals:
        if col in group.columns:
            ax.plot(
                group["elapsed_sec"],
                group[col],
                label=col,
                linewidth=0.9,
            )

    event_rows = group[group["Feedback"] == 1]

    for event_time in event_rows["elapsed_sec"]:
        ax.axvline(event_time, color="red", alpha=0.2, linewidth=0.7)

    ax.set_title(f"{recording_id}: Main signal traces")
    ax.set_xlabel("Elapsed time (seconds)")
    ax.set_ylabel("Recorded signal value")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    path = OUTPUT_DIR / f"{recording_id}_main_signals.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)

    # Plot 2: _a fields, which need extra validation.
    fig, ax = plt.subplots(figsize=(13, 5))

    for col in a_signals:
        if col in group.columns:
            ax.plot(
                group["elapsed_sec"],
                group[col],
                label=col,
                linewidth=0.8,
            )

    ax.set_title(f"{recording_id}: _a signal traces")
    ax.set_xlabel("Elapsed time (seconds)")
    ax.set_ylabel("Recorded value")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    path = OUTPUT_DIR / f"{recording_id}_a_signals.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)

print("\nPlots saved to:")
print(OUTPUT_DIR)
print("\nOpen the PNG files and inspect Dataset1 through Dataset7.")
