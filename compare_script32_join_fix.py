import pandas as pd
from pathlib import Path

folder = Path("data/processed/script32")
old = pd.read_csv(folder / "policy_summary_pre_join_fix.csv")
new = pd.read_csv(folder / "policy_summary.csv")

metrics = [
    "observed_crossing_events",
    "connected_at_crossing",
    "crossing_connectivity_pct",
    "connectivity_availability_pct",
    "simulated_outage_sec",
    "total_dispatches",
    "relays_deployed",
    "arrival_before_crossing",
    "arrival_after_crossing",
    "false_alarms_complete_window",
    "censored_dispatches",
]

old = old.set_index("policy")
new = new.set_index("policy")

print("PRE-FIX vs CORRECTED RESULTS")
for policy in sorted(set(old.index) & set(new.index)):
    print(f"\n--- {policy.upper()} ---")
    for metric in metrics:
        if metric not in old.columns or metric not in new.columns:
            continue
        before = old.loc[policy, metric]
        after = new.loc[policy, metric]
        print(f"{metric}: {before} -> {after} (change: {after - before:+.4f})")
