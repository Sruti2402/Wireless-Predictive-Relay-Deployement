from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "forecast_dataset.csv"
RF_RESULTS_PATH = ROOT / "data" / "processed" / "forecast_model_results.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "forecast_baseline_comparison.csv"

SIGNALS = [
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

HORIZONS = [1, 3, 5]


def main():
    df = pd.read_csv(DATA_PATH)
    rf_results = pd.read_csv(RF_RESULTS_PATH)

    rows = []

    for horizon in HORIZONS:
        for recording_id, group in df.groupby("recording_id"):
            for signal in SIGNALS:
                current = group[f"current_{signal}"].to_numpy(dtype=float)
                previous = group[f"{signal}_lag_1s"].to_numpy(dtype=float)
                actual = group[
                    f"future_{signal}_{horizon}s"
                ].to_numpy(dtype=float)

                # Baseline 1: assume the signal stays unchanged.
                persistence = current

                # Baseline 2: continue the recent approximately
                # one-second signal change into the future.
                recent_change = current - previous
                trend = current + horizon * recent_change

                for method, prediction in [
                    ("Persistence", persistence),
                    ("Linear trend", trend),
                ]:
                    rows.append({
                        "recording_id": recording_id,
                        "horizon_seconds": horizon,
                        "signal": signal,
                        "method": method,
                        "MAE": mean_absolute_error(actual, prediction),
                        "RMSE": np.sqrt(
                            mean_squared_error(actual, prediction)
                        ),
                        "test_rows": len(group),
                    })

    comparison = pd.DataFrame(rows)

    # Include Random Forest metrics from the previous experiment.
    rf = rf_results[
        [
            "test_recording",
            "horizon_seconds",
            "signal",
            "model_MAE",
            "model_RMSE",
            "test_rows",
        ]
    ].copy()

    rf = rf.rename(columns={
        "test_recording": "recording_id",
        "model_MAE": "MAE",
        "model_RMSE": "RMSE",
    })
    rf["method"] = "Random Forest"

    comparison = pd.concat(
        [comparison, rf[comparison.columns]],
        ignore_index=True,
    )

    # Weighted summary across held-out recordings.
    summary = (
        comparison.groupby(
            ["horizon_seconds", "method"], as_index=False
        )
        .apply(
            lambda g: pd.Series({
                "weighted_MAE": np.average(
                    g["MAE"], weights=g["test_rows"]
                ),
                "weighted_RMSE": np.sqrt(
                    np.average(
                        g["RMSE"] ** 2,
                        weights=g["test_rows"],
                    )
                ),
            }),
            include_groups=False,
        )
        .reset_index(drop=True)
    )

    comparison.to_csv(OUTPUT_PATH, index=False)

    print("=" * 72)
    print("FORECAST BASELINE COMPARISON")
    print("=" * 72)
    print(
        summary.sort_values(
            ["horizon_seconds", "weighted_MAE"]
        ).round(3).to_string(index=False)
    )

    print("\nBest method by horizon (weighted MAE):")
    for horizon in HORIZONS:
        subset = summary[summary["horizon_seconds"] == horizon]
        best = subset.loc[subset["weighted_MAE"].idxmin()]

        print(
            f"{horizon}s: {best['method']} "
            f"(MAE={best['weighted_MAE']:.3f})"
        )

    print("\nSaved detailed comparison:", OUTPUT_PATH)
    print("Comparison complete.")


if __name__ == "__main__":
    main()
