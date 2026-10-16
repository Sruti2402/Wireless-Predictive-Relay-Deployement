import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    mean_absolute_error,
    brier_score_loss,
)

INPUT = Path("data/processed/script25/dispatch_training_data.csv")
OUT = Path("data/processed/script29")
OUT.mkdir(parents=True, exist_ok=True)

HORIZON = 120.0
TRAIN_CAP_PER_RECORDING = 30000
SEED = 42

FEATURES = [
    "signal_quality", "signal_quality_smooth",
    "signal_5s_change", "signal_5s_decline",
    "speed_mps", "distance_from_base",
    "UL_level", "UR_level", "LL_level", "LR_level", "C_level",
    "signal_threshold", "relay_speed_mps",
    "setup_time_sec", "reconnect_time_sec",
    "dispatch_distance_m", "travel_time_sec",
    "deployment_time_sec", "safety_margin_sec",
]

def cap_rows(frame, n, seed):
    if len(frame) <= n:
        return frame.copy()
    return frame.sample(n=n, random_state=seed)

def main():
    df = pd.read_csv(INPUT)
    required = FEATURES + [
        "recording_id", "elapsed_sec",
        "time_to_next_low_signal_sec",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df[FEATURES] = df[FEATURES].replace(
        [np.inf, -np.inf], np.nan
    )
    df = df.dropna(subset=FEATURES).copy()

    target = "time_to_next_low_signal_sec"
    recording_end = df.groupby("recording_id")[
        "elapsed_sec"
    ].transform("max")
    remaining = recording_end - df["elapsed_sec"]

    observed = df[target].notna()
    within_horizon = observed & (df[target] >= 0) & (df[target] <= HORIZON)
    observed_after_horizon = observed & (df[target] > HORIZON)

    # A missing crossing is a known negative only when at least 120
    # seconds of recording remain. Otherwise the label is censored.
    known_no_crossing = (
        observed_after_horizon
        | (df[target].isna() & (remaining >= HORIZON))
    )
    known = within_horizon | known_no_crossing

    df["crossing_within_120s"] = np.nan
    df.loc[within_horizon, "crossing_within_120s"] = 1
    df.loc[known_no_crossing, "crossing_within_120s"] = 0

    data = df.loc[known].copy()
    recordings = sorted(data["recording_id"].unique())

    print(f"Source rows: {len(df):,}")
    print(f"Classifier-labelled rows: {len(data):,}")
    print(f"Censored/unlabelled rows: {len(df) - len(data):,}")
    print(f"Crossing within 120 seconds: "
          f"{int(data['crossing_within_120s'].sum()):,}")
    print(f"Recordings: {recordings}")

    if len(recordings) < 2:
        raise ValueError("Need at least two recordings.")

    predictions = []
    metrics = []

    for fold, held_out in enumerate(recordings, start=1):
        train = data[data["recording_id"] != held_out]
        # Predict over the full held-out recording, including censored rows.
        test = df[df["recording_id"] == held_out].copy()

        train = pd.concat([
            cap_rows(g, TRAIN_CAP_PER_RECORDING, SEED + fold)
            for _, g in train.groupby("recording_id")
        ], ignore_index=True)

        classifier = HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.08,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=SEED,
        )
        classifier.fit(
            train[FEATURES],
            train["crossing_within_120s"].astype(int),
        )
        probability = classifier.predict_proba(test[FEATURES])[:, 1]

        # Regress only on training examples whose crossing time is known
        # and lies inside the prediction horizon.
        reg_train = train.loc[
            train[target].notna()
            & (train[target] >= 0)
            & (train[target] <= HORIZON)
        ]
        if len(reg_train) < 100:
            raise ValueError(
                f"Too few regression examples in fold {held_out}"
            )

        regressor = HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.08,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=SEED,
        )
        regressor.fit(reg_train[FEATURES], reg_train[target])

        predicted_time = np.clip(
            regressor.predict(test[FEATURES]), 0, HORIZON
        )

        test["crossing_probability_120s"] = probability
        test["predicted_time_to_crossing_sec"] = predicted_time
        test["fold"] = fold
        predictions.append(test)

        labelled_test = test[
            test["crossing_within_120s"].notna()
        ].copy()
        y = labelled_test["crossing_within_120s"].astype(int).to_numpy()
        p = labelled_test["crossing_probability_120s"].to_numpy()
        row = {
            "held_out_recording": held_out,
            "test_rows": len(labelled_test),
            "full_trajectory_rows": len(test),
            "actual_crossing_rate": y.mean(),
            "average_precision": average_precision_score(y, p),
            "brier_score": brier_score_loss(y, p),
        }
        row["roc_auc"] = (
            roc_auc_score(y, p)
            if len(np.unique(y)) == 2 else np.nan
        )

        positive_test = labelled_test.loc[
            labelled_test[target].between(0, HORIZON)
        ]
        if len(positive_test):
            row["conditional_time_mae_sec"] = mean_absolute_error(
                positive_test[target],
                positive_test["predicted_time_to_crossing_sec"],
            )
        else:
            row["conditional_time_mae_sec"] = np.nan

        metrics.append(row)
        print(
            f"{held_out}: rows={len(test):,}, "
            f"AP={row['average_precision']:.3f}, "
            f"AUC={row['roc_auc']:.3f}, "
            f"Brier={row['brier_score']:.3f}, "
            f"conditional time MAE="
            f"{row['conditional_time_mae_sec']:.2f}s"
        )

    pred = pd.concat(predictions, ignore_index=True)
    metric_df = pd.DataFrame(metrics)

    pred.to_csv(OUT / "two_stage_oof_predictions.csv", index=False)
    metric_df.to_csv(OUT / "fold_metrics.csv", index=False)

    print("\nFOLD METRICS")
    print(metric_df.round(4).to_string(index=False))
    print("\nMEAN ACROSS HELD-OUT RECORDINGS")
    print(
        metric_df.drop(columns=["held_out_recording"])
        .mean(numeric_only=True).round(4).to_string()
    )
    print(f"\nSaved to: {OUT.resolve()}")
    print("These targets represent observed signal crossings,")
    print("not confirmed real-world communication outages.")

if __name__ == "__main__":
    main()
