
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

INPUT = Path("data/processed/script25/dispatch_training_data.csv")
OUT = Path("data/processed/script28")
OUT.mkdir(parents=True, exist_ok=True)

TARGET = "time_to_next_low_signal_sec"
MAX_HORIZON_SEC = 120.0
SAMPLE_PER_RECORDING = 25000
RANDOM_STATE = 42

FEATURES = [
    "signal_quality",
    "signal_quality_smooth",
    "signal_5s_change",
    "signal_5s_decline",
    "speed_mps",
    "distance_from_base",
    "UL_level",
    "UR_level",
    "LL_level",
    "LR_level",
    "C_level",
    "signal_threshold",
    "relay_speed_mps",
    "setup_time_sec",
    "reconnect_time_sec",
    "dispatch_distance_m",
    "travel_time_sec",
    "deployment_time_sec",
    "safety_margin_sec",
]

def sample_recording(group, limit, seed):
    if len(group) <= limit:
        return group.copy()
    return group.sample(n=limit, random_state=seed)

def metrics(y_true, y_pred):
    return {
        "n": len(y_true),
        "mae_sec": mean_absolute_error(y_true, y_pred),
        "rmse_sec": np.sqrt(mean_squared_error(y_true, y_pred)),
        "median_absolute_error_sec": float(
            np.median(np.abs(y_true - y_pred))
        ),
        "r2": r2_score(y_true, y_pred),
        "within_5_sec_pct": float(
            (np.abs(y_true - y_pred) <= 5).mean() * 100
        ),
        "within_10_sec_pct": float(
            (np.abs(y_true - y_pred) <= 10).mean() * 100
        ),
    }

def main():
    print("Loading dispatch training data...")
    df = pd.read_csv(INPUT)

    required = FEATURES + [TARGET, "recording_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Missing future-event times are censored/unknown, not infinite times.
    # Focus on crossings observed within the next 120 seconds.
    data = df.loc[
        df[TARGET].notna()
        & (df[TARGET] >= 0)
        & (df[TARGET] <= MAX_HORIZON_SEC)
    ].copy()

    data[FEATURES] = data[FEATURES].replace(
        [np.inf, -np.inf], np.nan
    )
    data = data.dropna(subset=FEATURES + [TARGET])

    recordings = sorted(data["recording_id"].unique())
    if len(recordings) < 2:
        raise ValueError("At least two recordings are required.")

    print(f"Rows in source: {len(df):,}")
    print(f"Rows with observed crossing within {MAX_HORIZON_SEC:.0f}s: "
          f"{len(data):,}")
    print(f"Recordings: {recordings}")
    print(f"Target median: {data[TARGET].median():.2f}s")
    print(f"Target mean: {data[TARGET].mean():.2f}s")

    fold_results = []
    prediction_frames = []

    # Leave one entire recording out for testing.
    for fold_num, held_out in enumerate(recordings, start=1):
        train_parts = []
        for recording in recordings:
            if recording == held_out:
                continue
            part = data[data["recording_id"] == recording]
            train_parts.append(
                sample_recording(part, SAMPLE_PER_RECORDING,
                                 RANDOM_STATE + fold_num)
            )

        train = pd.concat(train_parts, ignore_index=True)
        test = data[data["recording_id"] == held_out].copy()

        # Limit test size for practical runtime; this is still a recording-
        # held-out test set, not a random split of the same recording.
        test = sample_recording(
            test, SAMPLE_PER_RECORDING, RANDOM_STATE + 100 + fold_num
        )

        X_train = train[FEATURES]
        y_train = train[TARGET]
        X_test = test[FEATURES]
        y_test = test[TARGET]

        model = HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.08,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        )
        model.fit(X_train, y_train)
        pred = np.clip(model.predict(X_test), 0, MAX_HORIZON_SEC)

        result = metrics(y_test.to_numpy(), pred)
        result["held_out_recording"] = held_out
        result["train_rows"] = len(train)
        fold_results.append(result)

        fold_pred = test[
            ["recording_id", "elapsed_sec", "signal_threshold",
             "relay_speed_mps", "deployment_time_sec",
             "safety_margin_sec", TARGET]
        ].copy()
        fold_pred["predicted_time_to_crossing_sec"] = pred
        fold_pred["absolute_error_sec"] = np.abs(
            fold_pred[TARGET] - pred
        )
        fold_pred["fold"] = fold_num
        prediction_frames.append(fold_pred)

        print(
            f"Fold {fold_num}/{len(recordings)} | held out {held_out} | "
            f"n={result['n']:,} | MAE={result['mae_sec']:.2f}s | "
            f"RMSE={result['rmse_sec']:.2f}s | "
            f"within 10s={result['within_10_sec_pct']:.1f}%"
        )

    fold_df = pd.DataFrame(fold_results)
    pred_df = pd.concat(prediction_frames, ignore_index=True)

    fold_df.to_csv(OUT / "fold_metrics.csv", index=False)
    pred_df.to_csv(OUT / "oof_predictions.csv", index=False)

    print("\nFOLD METRICS")
    print(fold_df.round(3).to_string(index=False))

    print("\nMEAN ACROSS HELD-OUT RECORDINGS")
    metric_cols = [
        "mae_sec", "rmse_sec", "median_absolute_error_sec",
        "r2", "within_5_sec_pct", "within_10_sec_pct"
    ]
    print(fold_df[metric_cols].mean().round(3).to_string())

    # Simple baseline: always predict the training-fold median.
    baseline_rows = []
    for held_out in recordings:
        train = data[data["recording_id"] != held_out]
        test = data[data["recording_id"] == held_out]
        test = sample_recording(test, SAMPLE_PER_RECORDING,
                                RANDOM_STATE + 200)
        baseline_pred = np.full(len(test), train[TARGET].median())
        m = metrics(test[TARGET].to_numpy(), baseline_pred)
        m["held_out_recording"] = held_out
        baseline_rows.append(m)

    baseline_df = pd.DataFrame(baseline_rows)
    baseline_df.to_csv(OUT / "median_baseline_metrics.csv", index=False)

    print("\nMEDIAN BASELINE (MEAN ACROSS HELD-OUT RECORDINGS)")
    print(baseline_df[metric_cols].mean().round(3).to_string())

    print(f"\nSaved results to: {OUT.resolve()}")
    print("Important: this predicts an observed signal crossing,")
    print("not verified communication failure or real relay success.")

if __name__ == "__main__":
    main()
