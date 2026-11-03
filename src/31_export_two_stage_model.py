
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "script25" / "dispatch_training_data.csv"
OUT = ROOT / "models"
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
    print("Loading existing training data...")
    df = pd.read_csv(INPUT)

    required = FEATURES + [
        "recording_id",
        "elapsed_sec",
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
    within_horizon = (
        observed
        & (df[target] >= 0)
        & (df[target] <= HORIZON)
    )
    observed_after_horizon = observed & (df[target] > HORIZON)

    known_no_crossing = (
        observed_after_horizon
        | (df[target].isna() & (remaining >= HORIZON))
    )
    known = within_horizon | known_no_crossing

    df["crossing_within_120s"] = np.nan
    df.loc[within_horizon, "crossing_within_120s"] = 1
    df.loc[known_no_crossing, "crossing_within_120s"] = 0

    classifier_data = df.loc[known].copy()

    # Match the per-recording training cap used in Script 29.
    classifier_data = pd.concat(
        [
            cap_rows(group, TRAIN_CAP_PER_RECORDING, SEED + i)
            for i, (_, group) in enumerate(
                classifier_data.groupby("recording_id"), start=1
            )
        ],
        ignore_index=True,
    )

    regression_data = classifier_data.loc[
        classifier_data[target].notna()
        & classifier_data[target].between(0, HORIZON)
    ].copy()

    if classifier_data["crossing_within_120s"].nunique() != 2:
        raise ValueError("Classifier training requires both classes.")

    if len(regression_data) < 100:
        raise ValueError("Insufficient regression training examples.")

    print(f"Classifier rows: {len(classifier_data):,}")
    print(f"Regression rows: {len(regression_data):,}")
    print("Fitting classifier...")

    classifier = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=SEED,
    )
    classifier.fit(
        classifier_data[FEATURES],
        classifier_data["crossing_within_120s"].astype(int),
    )

    print("Fitting time-to-crossing regressor...")

    regressor = HistGradientBoostingRegressor(
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=SEED,
    )
    regressor.fit(
        regression_data[FEATURES],
        regression_data[target],
    )

    classifier_path = OUT / "crossing_classifier.joblib"
    regressor_path = OUT / "crossing_time_regressor.joblib"

    joblib.dump(classifier, classifier_path)
    joblib.dump(regressor, regressor_path)

    metadata = {
        "model_type": "two_stage_signal_crossing_prediction",
        "prediction_horizon_seconds": HORIZON,
        "features": FEATURES,
        "classifier": {
            "file": classifier_path.name,
            "algorithm": "HistGradientBoostingClassifier",
            "max_iter": 150,
            "learning_rate": 0.08,
            "max_leaf_nodes": 15,
            "l2_regularization": 1.0,
            "random_state": SEED,
        },
        "regressor": {
            "file": regressor_path.name,
            "algorithm": "HistGradientBoostingRegressor",
            "max_iter": 150,
            "learning_rate": 0.08,
            "max_leaf_nodes": 15,
            "l2_regularization": 1.0,
            "random_state": SEED,
        },
        "training": {
            "source": "data/processed/script25/dispatch_training_data.csv",
            "recordings": sorted(
                classifier_data["recording_id"].unique().tolist()
            ),
            "classifier_rows": len(classifier_data),
            "regression_rows": len(regression_data),
            "per_recording_cap": TRAIN_CAP_PER_RECORDING,
        },
        "validation_note": (
            "Deployment models fitted on all eligible recordings. "
            "Performance must be reported from the separate "
            "leave-one-recording-out evaluation, not training fit."
        ),
        "target_limitation": (
            "Predicts observed low-signal threshold crossings, "
            "not verified real-world communication outages."
        ),
    }

    metadata_path = OUT / "model_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nMODEL EXPORT COMPLETE")
    print(classifier_path)
    print(regressor_path)
    print(metadata_path)


if __name__ == "__main__":
    main()
