
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

INPUT_FILE = DATA_DIR / "future_risk_labels.csv"
METRICS_OUTPUT = DATA_DIR / "risk_model_metrics.csv"
PREDICTIONS_OUTPUT = DATA_DIR / "risk_model_oof_predictions.csv"

THRESHOLDS = [20, 30, 40]
PREDICTION_HORIZON_SEC = 30

RANDOM_STATE = 42

# Features available at the current time only.
SIGNAL_COLUMNS = [
    "UL_level",
    "UR_level",
    "LL_level",
    "LR_level",
    "C_level",
]

FEATURE_COLUMNS = (
    SIGNAL_COLUMNS
    + [f"{column}_smoothed" for column in SIGNAL_COLUMNS]
    + ["signal_metric"]
)


# ============================================================
# MODELS
# ============================================================

def make_models():
    return {
        "majority_baseline": DummyClassifier(
            strategy="most_frequent"
        ),

        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=RANDOM_STATE,
            )),
        ]),

        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=200,
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
    }


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, predictions, probabilities):
    y_true = np.asarray(y_true, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)

    try:
        roc_auc = (
            roc_auc_score(y_true, probabilities)
            if len(np.unique(y_true)) == 2
            else np.nan
        )
    except ValueError:
        roc_auc = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        average_precision = average_precision_score(
            y_true, probabilities
        )

    return {
        "sample_count": len(y_true),
        "positive_count": int(y_true.sum()),
        "positive_percent": 100 * float(y_true.mean()),
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, predictions
        ),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_true, predictions, zero_division=0
        ),
        "roc_auc": roc_auc,
        "average_precision": average_precision,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}\n"
            "Run src/17_build_future_risk_labels.py first."
        )

    df = pd.read_csv(INPUT_FILE)

    required = [
        "recording_id",
        "elapsed_sec",
        "hypothetical_threshold",
        "future_horizon_sec",
        "risk_label",
        *FEATURE_COLUMNS,
    ]

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Restrict to the selected prediction horizon.
    df = df[
        df["future_horizon_sec"] == PREDICTION_HORIZON_SEC
    ].copy()

    df["risk_label"] = pd.to_numeric(
        df["risk_label"], errors="coerce"
    )
    df = df.dropna(
        subset=["recording_id", "risk_label"]
    ).copy()
    df["risk_label"] = df["risk_label"].astype(int)

    # Do not include future_minimum_metric, time_to_crossing_sec,
    # future_horizon_sec, or any other future-derived fields as features.
    all_metrics = []
    all_predictions = []

    print("=" * 78)
    print("RECORDING-LEVEL RISK MODEL EVALUATION")
    print("=" * 78)
    print(f"Prediction horizon: {PREDICTION_HORIZON_SEC} seconds")
    print("Validation: leave one entire recording out")
    print(f"Features: {FEATURE_COLUMNS}")

    for threshold in THRESHOLDS:
        subset = df[
            df["hypothetical_threshold"] == threshold
        ].copy()

        recordings = sorted(
            subset["recording_id"].unique(),
            key=str,
        )

        print("\n" + "-" * 78)
        print(f"THRESHOLD: {threshold}")
        print(f"Rows: {len(subset):,}")
        print(f"Recordings: {len(recordings)}")
        print(
            "Positive labels: "
            f"{int(subset['risk_label'].sum()):,} "
            f"({100 * subset['risk_label'].mean():.2f}%)"
        )

        if len(recordings) < 2:
            print("Skipping: at least two recordings are required.")
            continue

        threshold_predictions = []

        for held_out_recording in recordings:
            train = subset[
                subset["recording_id"] != held_out_recording
            ]
            test = subset[
                subset["recording_id"] == held_out_recording
            ].copy()

            if train.empty or test.empty:
                continue

            X_train = train[FEATURE_COLUMNS].replace(
                [np.inf, -np.inf], np.nan
            )
            X_test = test[FEATURE_COLUMNS].replace(
                [np.inf, -np.inf], np.nan
            )

            y_train = train["risk_label"].to_numpy(dtype=int)
            y_test = test["risk_label"].to_numpy(dtype=int)

            for model_name, model in make_models().items():
                model.fit(X_train, y_train)

                predicted = model.predict(X_test).astype(int)

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(X_test)[:, 1]
                else:
                    probabilities = predicted.astype(float)

                fold_metrics = calculate_metrics(
                    y_test,
                    predicted,
                    probabilities,
                )

                # Keep fold-level metrics so we can see whether
                # performance is consistent across recordings.
                all_metrics.append({
                    "threshold": threshold,
                    "horizon_sec": PREDICTION_HORIZON_SEC,
                    "model": model_name,
                    "held_out_recording": held_out_recording,
                    **fold_metrics,
                })

                fold_predictions = pd.DataFrame({
                    "recording_id": test["recording_id"].to_numpy(),
                    "elapsed_sec": test["elapsed_sec"].to_numpy(),
                    "threshold": threshold,
                    "horizon_sec": PREDICTION_HORIZON_SEC,
                    "model": model_name,
                    "actual": y_test,
                    "predicted": predicted,
                    "predicted_probability": probabilities,
                })

                threshold_predictions.append(fold_predictions)
                all_predictions.append(fold_predictions)

            print(
                f"Tested held-out recording: {held_out_recording}"
            )

        if threshold_predictions:
            threshold_predictions_df = pd.concat(
                threshold_predictions,
                ignore_index=True,
            )

            print("\nOut-of-fold results:")
            for model_name in [
                "majority_baseline",
                "logistic_regression",
                "random_forest",
            ]:
                model_rows = threshold_predictions_df[
                    threshold_predictions_df["model"] == model_name
                ]

                metrics = calculate_metrics(
                    model_rows["actual"],
                    model_rows["predicted"],
                    model_rows["predicted_probability"],
                )

                print(
                    f"{model_name:24s} "
                    f"F1={metrics['f1']:.3f} | "
                    f"Recall={metrics['recall']:.3f} | "
                    f"Precision={metrics['precision']:.3f} | "
                    f"AP={metrics['average_precision']:.3f} | "
                    f"ROC-AUC={metrics['roc_auc']:.3f}"
                )

    # Save fold-level metrics.
    if not all_metrics:
        raise RuntimeError(
            "No model evaluations completed. Check labels and recordings."
        )

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(METRICS_OUTPUT, index=False)

    # Save out-of-fold predictions for later error analysis.
    predictions_df = pd.concat(
        all_predictions, ignore_index=True
    )
    predictions_df.to_csv(PREDICTIONS_OUTPUT, index=False)

    print("\n" + "=" * 78)
    print("EVALUATION COMPLETE")
    print("=" * 78)
    print(f"Fold metrics: {METRICS_OUTPUT}")
    print(f"Out-of-fold predictions: {PREDICTIONS_OUTPUT}")

    print(
        "\nInterpretation: this model predicts a hypothetical signal "
        "threshold crossing, not verified communication failure."
    )
    print(
        "Use average precision, recall, precision, and fold-to-fold "
        "consistency alongside ROC-AUC. Accuracy alone is insufficient."
    )


if __name__ == "__main__":
    main()
