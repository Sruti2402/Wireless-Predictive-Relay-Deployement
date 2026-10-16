
from pathlib import Path

import numpy as np
import pandas as pd

INPUT = Path("data/processed/script26/dispatch_oof_predictions.csv")
OUTPUT_DIR = Path("data/processed/script27")

PROBABILITY_CUTOFFS = [0.30, 0.40, 0.50, 0.60, 0.70]
DISPATCH_DISTANCE_M = 5.0
SAFETY_MARGIN_SEC = 2.0
ALERT_COOLDOWN_SEC = 30.0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

required = [
    "recording_id",
    "elapsed_sec",
    "signal_threshold",
    "relay_speed_mps",
    "setup_time_sec",
    "reconnect_time_sec",
    "signal_quality_smooth",
    "dispatch_now",
    "risk_probability",
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

numeric_cols = [
    "elapsed_sec",
    "signal_threshold",
    "relay_speed_mps",
    "setup_time_sec",
    "reconnect_time_sec",
    "signal_quality_smooth",
    "dispatch_now",
    "risk_probability",
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=numeric_cols + ["recording_id"]).copy()
df = df[df["relay_speed_mps"] > 0].copy()

print("=" * 72)
print("SCRIPT 27 — EVENT-BASED DISPATCH POLICY EVALUATION")
print("=" * 72)
print(f"Rows loaded: {len(df):,}")
print(f"Recordings: {df['recording_id'].nunique()}")
print("\nColumn types:")
print(df[
    [
        "relay_speed_mps",
        "setup_time_sec",
        "reconnect_time_sec",
        "signal_quality_smooth",
    ]
].dtypes)

results = []
detail_rows = []

group_cols = [
    "recording_id",
    "signal_threshold",
    "relay_speed_mps",
    "setup_time_sec",
    "reconnect_time_sec",
]

for group_key, g in df.groupby(group_cols, sort=False):
    recording, threshold, relay_speed, setup, reconnect = group_key

    g = g.sort_values("elapsed_sec").reset_index(drop=True)

    t = g["elapsed_sec"].to_numpy(dtype=float)
    signal = g["signal_quality_smooth"].to_numpy(dtype=float)
    prob = g["risk_probability"].to_numpy(dtype=float)

    if len(g) < 2:
        continue

    # A crossing is a transition from above the threshold to
    # at or below it. Consecutive low-signal samples count once.
    crossing_idx = np.flatnonzero(
        (signal[:-1] > float(threshold))
        & (signal[1:] <= float(threshold))
    ) + 1
    crossing_times = t[crossing_idx]

    travel_time = DISPATCH_DISTANCE_M / float(relay_speed)
    deployment_time = travel_time + float(setup) + float(reconnect)
    latest_acceptable_lead = deployment_time + SAFETY_MARGIN_SEC

    # A reactive policy dispatches at the crossing itself.
    # It therefore has no lead time to complete deployment.
    results.append({
        "recording_id": recording,
        "signal_threshold": threshold,
        "relay_speed_mps": relay_speed,
        "setup_time_sec": setup,
        "reconnect_time_sec": reconnect,
        "policy": "ReactiveThreshold",
        "probability_cutoff": np.nan,
        "crossings": len(crossing_times),
        "alerts": len(crossing_times),
        "feasible_covered_crossings": 0,
        "coverage_pct": 0.0 if len(crossing_times) else np.nan,
        "false_alerts": 0,
        "alert_precision": np.nan,
        "median_lead_sec": 0.0 if len(crossing_times) else np.nan,
        "deployment_time_sec": deployment_time,
    })

    for cutoff in PROBABILITY_CUTOFFS:
        # Only trigger above the threshold. Cooldown prevents
        # a continuously high probability from generating alerts
        # at every sample.
        alert_times = []
        last_alert = -np.inf

        for i in range(len(g)):
            if signal[i] <= float(threshold):
                continue

            if prob[i] < cutoff:
                continue

            if t[i] - last_alert < ALERT_COOLDOWN_SEC:
                continue

            alert_times.append(t[i])
            last_alert = t[i]

        # Match alerts to crossings only when the warning lead
        # lies within the assumed deployment-feasibility window.
        used_alerts = set()
        matched_leads = []

        for crossing_time in crossing_times:
            candidates = []

            for alert_i, alert_time in enumerate(alert_times):
                if alert_i in used_alerts:
                    continue

                lead = crossing_time - alert_time

                if (
                    deployment_time <= lead
                    <= latest_acceptable_lead
                ):
                    candidates.append((alert_i, lead))

            if candidates:
                # Prefer the latest alert that still leaves
                # enough time for deployment.
                alert_i, lead = max(candidates, key=lambda x: x[1] * -1)
                used_alerts.add(alert_i)
                matched_leads.append(lead)

        covered = len(matched_leads)
        alert_count = len(alert_times)
        false_alerts = alert_count - len(used_alerts)

        result = {
            "recording_id": recording,
            "signal_threshold": threshold,
            "relay_speed_mps": relay_speed,
            "setup_time_sec": setup,
            "reconnect_time_sec": reconnect,
            "policy": "LearnedRiskModel",
            "probability_cutoff": cutoff,
            "crossings": len(crossing_times),
            "alerts": alert_count,
            "feasible_covered_crossings": covered,
            "coverage_pct": (
                100 * covered / len(crossing_times)
                if len(crossing_times) else np.nan
            ),
            "false_alerts": false_alerts,
            "alert_precision": (
                covered / alert_count if alert_count else np.nan
            ),
            "median_lead_sec": (
                float(np.median(matched_leads))
                if matched_leads else np.nan
            ),
            "deployment_time_sec": deployment_time,
        }

        results.append(result)

        detail_rows.append({
            **result,
            "matched_leads_sec": ";".join(
                f"{x:.3f}" for x in matched_leads
            ),
        })

results_df = pd.DataFrame(results)

# Aggregate by policy and scenario, weighted by total events/alerts.
summary_rows = []

summary_groups = [
    "signal_threshold",
    "relay_speed_mps",
    "setup_time_sec",
    "reconnect_time_sec",
    "policy",
    "probability_cutoff",
]

for keys, g in results_df.groupby(
    summary_groups, dropna=False, sort=True
):
    crossings = int(g["crossings"].sum())
    alerts = int(g["alerts"].sum())
    covered = int(g["feasible_covered_crossings"].sum())

    valid_leads = g["median_lead_sec"].dropna()

    summary_rows.append({
        **dict(zip(summary_groups, keys)),
        "crossings": crossings,
        "alerts": alerts,
        "feasible_covered_crossings": covered,
        "coverage_pct": 100 * covered / crossings if crossings else np.nan,
        "false_alerts": int(g["false_alerts"].sum()),
        "alert_precision": covered / alerts if alerts else np.nan,
        "median_group_lead_sec": (
            float(valid_leads.median()) if len(valid_leads) else np.nan
        ),
        "deployment_time_sec": float(g["deployment_time_sec"].median()),
    })

summary = pd.DataFrame(summary_rows)

summary_path = OUTPUT_DIR / "policy_summary.csv"
detail_path = OUTPUT_DIR / "policy_recording_details.csv"

summary.to_csv(summary_path, index=False)
results_df.to_csv(detail_path, index=False)

print("\n" + "=" * 72)
print("TOP LEARNED-POLICY CONFIGURATIONS")
print("=" * 72)

learned = summary[summary["policy"] == "LearnedRiskModel"].copy()

if not learned.empty:
    learned = learned.sort_values(
        ["coverage_pct", "alert_precision"],
        ascending=[False, False],
    )

    print(learned.head(20).to_string(index=False))

print("\nSaved:")
print(summary_path.resolve())
print(detail_path.resolve())

print(
    "\nInterpretation: these are offline event-matching results "
    "under assumed travel/setup/reconnect times. They do not prove "
    "that a physical relay restores communication."
)
