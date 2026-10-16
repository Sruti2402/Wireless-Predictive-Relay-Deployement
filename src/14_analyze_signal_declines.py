
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "combined_repaired.csv"
OUTPUT_DIR = ROOT / "data" / "processed"

df = pd.read_csv(INPUT)

SIGNALS = [
    "UL_level",
    "UR_level",
    "LL_level",
    "LR_level",
    "C_level",
]

required = ["recording_id", "elapsed_sec", "Feedback"] + SIGNALS
df = df[required].copy()

for col in ["elapsed_sec", "Feedback"] + SIGNALS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["recording_id", "elapsed_sec"])
df = df.sort_values(["recording_id", "elapsed_sec"])

# Avoid treating duplicate timestamps as separate observations.
df = df.drop_duplicates(
    subset=["recording_id", "elapsed_sec"],
    keep="first",
)

# Rolling mean smooths individual readings without changing the source data.
# Five seconds is approximately 24 samples at the observed sampling rate.
window_seconds = 5
decline_threshold = 10
minimum_channels = 3

recording_results = []
event_results = []

for recording_id, group in df.groupby("recording_id"):
    group = group.sort_values("elapsed_sec").copy()
    times = group["elapsed_sec"].to_numpy(dtype=float)

    if len(group) < 3:
        continue

    dt = np.diff(times)
    positive_dt = dt[dt > 0]

    if len(positive_dt) == 0:
        continue

    sample_interval = float(np.median(positive_dt))
    window_samples = max(
        2, int(round(window_seconds / sample_interval))
    )

    # Compare the current smoothed signal with its value
    # approximately five seconds earlier.
    decline_columns = []

    for signal in SIGNALS:
        smooth = group[signal].rolling(
            window=window_samples,
            min_periods=window_samples,
        ).mean()

        past_smooth = smooth.shift(window_samples)

        drop = past_smooth - smooth

        drop_column = f"{signal}_drop_5s"
        group[drop_column] = drop

        # Positive drop means the signal became lower.
        decline_columns.append(drop_column)

    drops = group[decline_columns]

    # Count channels with a decline of at least 10 recorded units.
    group["declining_channels"] = (
        drops >= decline_threshold
    ).sum(axis=1)

    # Exploratory rule only, NOT a validated link-failure threshold.
    group["multi_channel_decline"] = (
        group["declining_channels"] >= minimum_channels
    )

    active = group["multi_channel_decline"].fillna(False).to_numpy()
    event_mask = group["Feedback"].eq(1).to_numpy()

    # Summarize the percentage of usable samples with a joint decline.
    valid = group[decline_columns].notna().all(axis=1)

    valid_count = int(valid.sum())
    decline_count = int(
        (group.loc[valid, "multi_channel_decline"]).sum()
    )

    recording_results.append({
        "recording_id": recording_id,
        "rows": len(group),
        "duration_sec": round(times[-1] - times[0], 2),
        "valid_comparison_rows": valid_count,
        "joint_decline_rows": decline_count,
        "joint_decline_percent": (
            round(100 * decline_count / valid_count, 2)
            if valid_count else np.nan
        ),
        "feedback_events": int(event_mask.sum()),
    })

    # For each feedback event, find the closest preceding joint decline.
    decline_times = times[active & valid.to_numpy()]

    for idx in np.flatnonzero(event_mask):
        event_time = times[idx]

        prior_declines = decline_times[decline_times <= event_time]

        if len(prior_declines):
            last_decline = prior_declines[-1]
            elapsed_since_decline = event_time - last_decline
        else:
            last_decline = np.nan
            elapsed_since_decline = np.nan

        event_results.append({
            "recording_id": recording_id,
            "event_time_sec": event_time,
            "last_joint_decline_time_sec": last_decline,
            "seconds_since_joint_decline": elapsed_since_decline,
            "joint_decline_at_event": bool(active[idx]),
        })

recording_summary = pd.DataFrame(recording_results)
event_summary = pd.DataFrame(event_results)

recording_path = OUTPUT_DIR / "signal_decline_summary.csv"
event_path = OUTPUT_DIR / "feedback_decline_context.csv"

recording_summary.to_csv(recording_path, index=False)
event_summary.to_csv(event_path, index=False)

print("\nJOINT SIGNAL DECLINE SUMMARY")
print("=" * 75)
print(recording_summary.to_string(index=False))

print("\nFEEDBACK EVENT CONTEXT")
print("=" * 75)

if event_summary.empty:
    print("No Feedback=1 events found.")
else:
    print(event_summary.head(30).to_string(index=False))

    matched = event_summary[
        event_summary["seconds_since_joint_decline"].notna()
    ]

    print("\nEvents with a preceding joint decline:", len(matched))
    if len(matched):
        print(
            "Median seconds since last joint decline:",
            round(matched["seconds_since_joint_decline"].median(), 2),
        )

print("\nSaved:")
print(recording_path)
print(event_path)

print(
    "\nIMPORTANT: This is exploratory analysis. "
    "Feedback semantics and a real link-failure threshold "
    "have not been established."
)
