from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "forecast_dataset.csv"
OUTPUT = ROOT / "data" / "processed"

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

# Start with a reasonably sized model for a reproducible baseline.
N_ESTIMATORS = 100
RANDOM_STATE = 42


def calculate_metrics(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
    }


def main():
    df = pd.read_csv(INPUT)

    if "recording_id" not in df.columns:
        raise ValueError("recording_id is required for recording-wise evaluation.")

    # Only use information available at prediction time.
    # Explicitly exclude timestamps, recording identifiers, and future targets.
    feature_cols = [
        col for col in df.columns
        if (
            col.startswith("current_")
            or "_lag_" in col
            or col.startswith("displacement_")
            or col.startswith("velocity_x_")
            or col.startswith("velocity_y_")
            or col in ["robot_pos_x", "robot_pos_y"]
        )
    ]

    if not feature_cols:
        raise ValueError("No input feature columns were found.")

    df[feature_cols] = df[feature_cols].apply(
        pd.to_numeric, errors="coerce"
    )

    if df[feature_cols].isna().any().any():
        raise ValueError("Input features contain missing or invalid values.")

    recordings = sorted(df["recording_id"].unique())

    if len(recordings) < 2:
        raise ValueError("At least two recordings are needed.")

    print("=" * 72)
    print("RECORDING-WISE SIGNAL FORECAST EVALUATION")
    print("=" * 72)
    print(f"Rows: {len(df)}")
    print(f"Input features: {len(feature_cols)}")
    print(f"Recordings: {len(recordings)}")
    print(f"Recordings: {recordings}")
    print(f"Forecast horizons: {HORIZONS} seconds")
    print("\nEach recording is tested separately while all other recordings train the model.")

    results = []

    for horizon in HORIZONS:
        target_cols = [
            f"future_{signal}_{horizon}s"
            for signal in SIGNALS
        ]

        missing = set(target_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing targets: {sorted(missing)}")

        print(f"\n{'=' * 72}")
        print(f"FORECAST HORIZON: {horizon} SECOND(S)")
        print(f"{'=' * 72}")

        for test_recording in recordings:
            train = df[df["recording_id"] != test_recording]
            test = df[df["recording_id"] == test_recording]

            X_train = train[feature_cols]
            X_test = test[feature_cols]

            y_train = train[target_cols].to_numpy(dtype=float)
            y_test = test[target_cols].to_numpy(dtype=float)

            # Baseline: assume the current signal remains unchanged.
            baseline_cols = [
                f"current_{signal}" for signal in SIGNALS
            ]
            baseline = test[baseline_cols].to_numpy(dtype=float)

            # One multi-output model forecasts all ten signal measurements.
            model = RandomForestRegressor(
                n_estimators=N_ESTIMATORS,
                min_samples_leaf=3,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )

            model.fit(X_train, y_train)
            prediction = model.predict(X_test)

            for j, signal in enumerate(SIGNALS):
                baseline_metrics = calculate_metrics(
                    y_test[:, j], baseline[:, j]
                )
                model_metrics = calculate_metrics(
                    y_test[:, j], prediction[:, j]
                )

                baseline_mae = baseline_metrics["MAE"]
                model_mae = model_metrics["MAE"]

                improvement = (
                    100 * (baseline_mae - model_mae) / baseline_mae
                    if baseline_mae > 0
                    else np.nan
                )

                results.append({
                    "test_recording": test_recording,
                    "horizon_seconds": horizon,
                    "signal": signal,
                    "test_rows": len(test),
                    "baseline_MAE": baseline_mae,
                    "baseline_RMSE": baseline_metrics["RMSE"],
                    "model_MAE": model_mae,
                    "model_RMSE": model_metrics["RMSE"],
                    "MAE_improvement_percent": improvement,
                })

            print(
                f"Test recording: {test_recording:10s} | "
                f"train rows: {len(train):5d} | "
                f"test rows: {len(test):5d}"
            )

    detail = pd.DataFrame(results)

    detail_path = OUTPUT / "forecast_model_results.csv"
    detail.to_csv(detail_path, index=False)

    # Aggregate errors across held-out recordings, weighting each
    # recording by its number of evaluated rows.
    summary = (
        detail.groupby(["horizon_seconds", "signal"], as_index=False)
        .apply(
            lambda g: pd.Series({
                "baseline_MAE": np.average(
                    g["baseline_MAE"], weights=g["test_rows"]
                ),
                "model_MAE": np.average(
                    g["model_MAE"], weights=g["test_rows"]
                ),
                "baseline_RMSE": np.sqrt(
                    np.average(
                        g["baseline_RMSE"] ** 2,
                        weights=g["test_rows"],
                    )
                ),
                "model_RMSE": np.sqrt(
                    np.average(
                        g["model_RMSE"] ** 2,
                        weights=g["test_rows"],
                    )
                ),
            }),
            include_groups=False,
        )
        .reset_index(drop=True)
    )

    summary["MAE_improvement_percent"] = np.where(
        summary["baseline_MAE"] > 0,
        100 * (
            summary["baseline_MAE"] - summary["model_MAE"]
        ) / summary["baseline_MAE"],
        np.nan,
    )

    summary_path = OUTPUT / "forecast_model_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 72)
    print("SUMMARY: RANDOM FOREST VS PERSISTENCE BASELINE")
    print("=" * 72)

    print(
        summary.round(3).to_string(index=False)
    )

    print("\nOverall weighted comparison by horizon:")

    for horizon in HORIZONS:
        subset = detail[detail["horizon_seconds"] == horizon]

        baseline_mae = np.average(
            subset["baseline_MAE"], weights=subset["test_rows"]
        )
        model_mae = np.average(
            subset["model_MAE"], weights=subset["test_rows"]
        )

        improvement = (
            100 * (baseline_mae - model_mae) / baseline_mae
            if baseline_mae > 0
            else np.nan
        )

        print(
            f"{horizon}s | baseline MAE={baseline_mae:.3f} | "
            f"model MAE={model_mae:.3f} | "
            f"improvement={improvement:.2f}%"
        )

    print("\nSaved detailed results:", detail_path)
    print("Saved summary:", summary_path)
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
