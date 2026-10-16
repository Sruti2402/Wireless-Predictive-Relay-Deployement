import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

SCRIPT35_PATH = SRC / "35_compare_relay_placement.py"
OUTPUT_DIR = ROOT / "data" / "processed" / "script36"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RADIO_RANGES = [8.0, 10.0, 12.0]
POLICIES = ["predictive", "adaptive"]
PLACEMENT_STRATEGIES = ["current_position", "movement_aware"]

spec = importlib.util.spec_from_file_location(
    "simulator35", SCRIPT35_PATH
)
simulator35 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(simulator35)

simulator32 = simulator35.simulator32


def main():
    print("Running actual predictive vs adaptive simulations...")
    data = simulator32.load_data()

    grouped = data.groupby(
        simulator32.SCENARIO_KEYS,
        sort=False,
        dropna=False,
    )
    rows = []
    dispatch_rows = []
    total = (
        len(grouped)
        * len(RADIO_RANGES)
        * len(POLICIES)
        * len(PLACEMENT_STRATEGIES)
    )
    count = 0

    for _, trajectory in grouped:
        for radio_range in RADIO_RANGES:
            for policy in POLICIES:
                for placement in PLACEMENT_STRATEGIES:
                    result = simulator35.simulate(
                        trajectory.copy(),
                        policy=policy,
                        placement_strategy=placement,
                        radio_range=radio_range,
                        base=(0.0, 0.0),
                    )
                    dispatch_log = result.pop("dispatch_log")
                    rows.append(result)
                    dispatch_rows.extend(dispatch_log)
                    count += 1

        if count % 100 == 0:
            print(f"Completed {count}/{total} runs")

    details = pd.DataFrame(rows)
    dispatches = pd.DataFrame(dispatch_rows)
    details.to_csv(
        OUTPUT_DIR / "adaptive_policy_details.csv",
        index=False,
    )
    dispatches.to_csv(
        OUTPUT_DIR / "adaptive_dispatch_log.csv",
        index=False,
    )

    summary_rows = []
    group_columns = [
        "radio_range_m",
        "policy",
        "placement_strategy",
    ]

    for keys, group in details.groupby(group_columns, dropna=False):
        radio_range, policy, strategy = keys
        crossings = int(group["observed_crossing_events"].sum())
        connected_crossings = int(group["connected_at_crossing"].sum())
        total_time = float(group["total_observation_sec"].sum())
        connected_time = float(group["connected_time_sec"].sum())
        total_dispatches = int(group["total_dispatches"].sum())
        deployed = int(group["relays_deployed"].sum())

        summary_rows.append({
            "radio_range_m": radio_range,
            "policy": policy,
            "placement_strategy": strategy,
            "trajectories": len(group),
            "observed_crossing_events": crossings,
            "connected_at_crossing": connected_crossings,
            "crossing_connectivity_pct": (
                100 * connected_crossings / crossings
                if crossings else np.nan
            ),
            "connectivity_availability_pct": (
                100 * connected_time / total_time
                if total_time else np.nan
            ),
            "simulated_outage_sec": float(
                group["simulated_outage_sec"].sum()
            ),
            "simulated_outage_starts": int(
                group["simulated_outage_starts"].sum()
            ),
            "total_dispatches": total_dispatches,
            "relays_deployed": deployed,
            "deployed_per_dispatch_pct": (
                100 * deployed / total_dispatches
                if total_dispatches else np.nan
            ),
            "placement_failures": int(
                group["placement_failures"].sum()
            ),
            "arrivals_not_observed": int(
                group["arrivals_not_observed"].sum()
            ),
            "arrival_before_crossing": int(
                group["arrival_before_crossing"].sum()
            ),
            "arrival_after_crossing": int(
                group["arrival_after_crossing"].sum()
            ),
            "false_alarms_complete_window": int(
                group["false_alarms_complete_window"].sum()
            ),
            "censored_dispatches": int(
                group["censored_dispatches"].sum()
            ),
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        OUTPUT_DIR / "adaptive_policy_summary.csv",
        index=False,
    )

    print("\nPOOLED RESULTS")
    print(summary.round(2).to_string(index=False))
    print(f"\nOutput: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
