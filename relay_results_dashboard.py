"""Generate refreshable charts from existing relay simulation CSV outputs.

Run from the project root:
    python .\relay_results_dashboard.py

Each run re-reads the latest CSV files under data/processed/script32, script35,
and script36, then overwrites PNG charts and a combined summary CSV under
 dashboard/data/.
"""
from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed"
OUT = ROOT / "dashboard" / "data"
OUT.mkdir(parents=True, exist_ok=True)


def normalize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    aliases = {
        "radio_range": "radio_range_m",
        "radio_range_meters": "radio_range_m",
        "crossing_connectivity_percent": "crossing_connectivity_pct",
        "connectivity_availability_percent": "connectivity_availability_pct",
        "simulated_outage_seconds": "simulated_outage_sec",
        "dispatch_count": "total_dispatches",
        "total_dispatch_count": "total_dispatches",
        "placement": "placement_strategy",
    }
    return df.rename(columns={c: aliases[c] for c in df.columns if c in aliases})


def find_summary(folder, required, prefer=()):
    base = DATA / folder
    if not base.exists():
        return None, None
    candidates = []
    for path in base.rglob("*.csv"):
        try:
            # Read headers first; then only load small, matching summary files.
            header = normalize_columns(pd.read_csv(path, nrows=0))
            cols = set(header.columns)
            if required.issubset(cols):
                name = path.name.lower()
                score = sum(word in name for word in prefer)
                # Prefer aggregate tables; avoid row-level dispatch logs.
                if any(word in name for word in ("dispatch_log", "predictions", "training_data", "simulation_inputs")):
                    score -= 10
                candidates.append((score, path))
        except Exception:
            continue
    if not candidates:
        return None, None
    candidates.sort(key=lambda x: (x[0], x[1].stat().st_size), reverse=True)
    for _, path in candidates:
        try:
            df = normalize_columns(pd.read_csv(path))
            if len(df) and len(df) <= 5000:
                return path, df
        except Exception:
            pass
    return None, None


def save_bar(df, x, y, title, ylabel, filename, hue=None, sort=False, percent=False):
    if df is None or df.empty or x not in df or y not in df:
        return False
    plot = df.copy()
    plot[y] = pd.to_numeric(plot[y], errors="coerce")
    plot = plot.dropna(subset=[x, y])
    if plot.empty:
        return False
    if sort:
        plot = plot.sort_values(y)
    labels = plot[x].astype(str)
    if hue and hue in plot:
        labels = labels + " | " + plot[hue].astype(str)
    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(labels, plot[y])
    ax.set_title(title, fontsize=14, pad=14)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=.25)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelrotation=25)
    if percent:
        ax.set_ylim(0, max(100, float(plot[y].max()) * 1.08))
    for bar, value in zip(bars, plot[y]):
        text = f"{value:.2f}%" if percent else (f"{value:,.0f}" if abs(value) >= 100 else f"{value:.2f}")
        ax.annotate(text, (bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def save_tradeoff(df, filename, label_cols):
    if df is None or df.empty:
        return False
    need = {"total_dispatches", "simulated_outage_sec"}
    if not need.issubset(df.columns):
        return False
    plot = df.copy()
    for c in need:
        plot[c] = pd.to_numeric(plot[c], errors="coerce")
    plot = plot.dropna(subset=list(need))
    if plot.empty:
        return False
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(plot["total_dispatches"], plot["simulated_outage_sec"], s=70)
    for _, row in plot.iterrows():
        label = " / ".join(str(row[c]) for c in label_cols if c in plot.columns)
        ax.annotate(label, (row["total_dispatches"], row["simulated_outage_sec"]),
                    xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Total relay dispatches (count)")
    ax.set_ylabel("Simulated outage (seconds)")
    ax.set_title("Trade-off: Dispatch Count vs Simulated Outage")
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    generated = []
    summaries = []

    # Script 32: policy-level baseline comparison.
    p32, s32 = find_summary(
        "script32",
        {"policy", "simulated_outage_sec", "total_dispatches", "crossing_connectivity_pct"},
        prefer=("summary", "comparison", "metrics", "policy"),
    )
    if s32 is not None:
        s32["source_script"] = "script32"
        summaries.append(s32)
        for ok, name in [
            (save_bar(s32, "policy", "simulated_outage_sec", "Script 32 — Simulated Outage by Policy", "Simulated outage (seconds)", "01_script32_outage_by_policy.png"), "01_script32_outage_by_policy.png"),
            (save_bar(s32, "policy", "crossing_connectivity_pct", "Script 32 — Connectivity at Threshold Crossings", "Connected crossing events (%)", "02_script32_crossing_connectivity.png", percent=True), "02_script32_crossing_connectivity.png"),
            (save_bar(s32, "policy", "connectivity_availability_pct", "Script 32 — Overall Connectivity Availability", "Connectivity availability (%)", "03_script32_connectivity_availability.png", percent=True), "03_script32_connectivity_availability.png"),
            (save_bar(s32, "policy", "total_dispatches", "Script 32 — Relay Dispatch Count by Policy", "Total dispatches (count)", "04_script32_dispatch_count.png"), "04_script32_dispatch_count.png"),
            (save_tradeoff(s32, "05_script32_dispatch_vs_outage.png", ["policy"]), "05_script32_dispatch_vs_outage.png"),
        ]:
            if ok and name not in generated: generated.append(name)
        print(f"Script 32 source: {p32}")
    else:
        print("WARNING: Could not find Script 32 summary CSV with policy/outage/dispatch/connectivity columns.")

    # Script 35: placement strategy comparison, preserving radio range and policy.
    req35 = {"radio_range_m", "policy", "placement_strategy", "simulated_outage_sec", "crossing_connectivity_pct"}
    p35, s35 = find_summary("script35", req35, prefer=("summary", "comparison", "placement", "metrics"))
    if s35 is not None:
        s35["source_script"] = "script35"
        summaries.append(s35)
        s35["scenario"] = s35["policy"].astype(str) + " / " + s35["placement_strategy"].astype(str) + " / " + s35["radio_range_m"].astype(str) + "m"
        if save_bar(s35, "scenario", "simulated_outage_sec", "Script 35 — Placement & Policy: Simulated Outage", "Simulated outage (seconds)", "06_script35_placement_outage.png", sort=True): generated.append("06_script35_placement_outage.png")
        if save_bar(s35, "scenario", "crossing_connectivity_pct", "Script 35 — Placement & Policy: Crossing Connectivity", "Connected crossing events (%)", "07_script35_placement_connectivity.png", sort=True, percent=True): generated.append("07_script35_placement_connectivity.png")
        print(f"Script 35 source: {p35}")
    else:
        print("WARNING: Could not find Script 35 aggregate CSV with radio range, policy, placement, outage and connectivity columns.")

    # Script 36: adaptive vs predictive comparison.
    req36 = {"radio_range_m", "policy", "placement_strategy", "simulated_outage_sec", "crossing_connectivity_pct"}
    p36, s36 = find_summary("script36", req36, prefer=("summary", "comparison", "adaptive", "metrics"))
    if s36 is not None:
        s36["source_script"] = "script36"
        summaries.append(s36)
        s36["scenario"] = s36["policy"].astype(str) + " / " + s36["placement_strategy"].astype(str) + " / " + s36["radio_range_m"].astype(str) + "m"
        if save_bar(s36, "scenario", "simulated_outage_sec", "Script 36 — Adaptive vs Predictive: Simulated Outage", "Simulated outage (seconds)", "08_script36_adaptive_outage.png", sort=True): generated.append("08_script36_adaptive_outage.png")
        if save_bar(s36, "scenario", "crossing_connectivity_pct", "Script 36 — Adaptive vs Predictive: Crossing Connectivity", "Connected crossing events (%)", "09_script36_adaptive_connectivity.png", sort=True, percent=True): generated.append("09_script36_adaptive_connectivity.png")
        print(f"Script 36 source: {p36}")
    else:
        print("WARNING: Could not find Script 36 aggregate CSV with radio range, policy, placement, outage and connectivity columns.")

    if summaries:
        combined = pd.concat(summaries, ignore_index=True, sort=False)
        combined.to_csv(OUT / "dashboard_combined_summary.csv", index=False)
        print(f"Combined summary: {OUT / 'dashboard_combined_summary.csv'}")
    if not generated:
        print("\nNo charts generated. Please inspect CSV names/headers in data/processed/script32, script35 and script36.")
        sys.exit(2)
    print("\nCharts refreshed:")
    for name in generated:
        print(f"  {OUT / name}")
    print("\nRerun this script after each simulation to refresh the visuals.")

if __name__ == "__main__":
    main()
