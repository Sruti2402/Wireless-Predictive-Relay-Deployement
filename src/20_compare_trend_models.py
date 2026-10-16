
from pathlib import Path

import numpy as np
import pandas as pd

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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = (
    BASE_DIR / "data" / "processed"
    / "future_risk_labels_with_trends.csv"
)
OUTPUT_DIR = BASE_DIR / "data" / "processed"

HORIZON = 30
THRESHOLDS = [20, 30, 40]
RANDOM_STATE = 42

CHANNELS = ["UL", "UR", "LL", "LR", "C"]

BASE_FEATURES = (
    ["signal_metric"]
    + [f"{ch}_level" for ch in CHANNELS]
    + [f"{ch}_level_smoothed" for ch in CHANNELS]
)

TREND_FEATURES = (
    ["metric_change_1s", "metric_change_3s",
     "metric_change_5s", "metric_change_10s"]
    + [
        f"{ch}_change_{seconds}s"
        for seconds in [1, 3, 5, 10]
        for ch in CHANNELS
    ]
)


def make_model(model_name):
    if model_name == "logistic_regression":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        )

    if model_name == "random_forest":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=200,
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
        )

    raise ValueError(f"Unknown model: {model_name}")


def calculate_metrics(y_true, probabilities):
    predictions = (probabilities >= 0.5).astype(int)

    result = {
        "sample_count": len(y_true),
        "positive_count": int(np.sum(y_true)),
        "positive_percent": 100 * float(np.mean(y_true)),
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
        "average_precision": (
            average_precision_score(y_true, probabilities)
            if len(np.unique(y_true)) == 2
            else np.nan
        ),
        "roc_auc": (
            roc_auc_score(y_true, probabilities)
            if len(np.unique(y_true)) == 2
            else np.nan
        ),
    }
    return result


def main():
    print("=" * 78)
    print("BASELINE VS SIGNAL-TREND MODEL")
    print("=" * 78)

    df = pd.read_csv(INPUT_FILE)

    required = [
        "recording_id",
        "elapsed_sec",
        "hypothetical_threshold",
        "future_horizon_sec",
        "risk_label",
    ] + BASE_FEATURES + TREND_FEATURES

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[df["future_horizon_sec"] == HORIZON].copy()
    df = df[df["hypothetical_threshold"].isin(THRESHOLDS)].copy()

    # Fair comparison: both model types use exactly the same rows.
    # Remove initial timestamps lacking one or more trend features.
    df = df.dropna(subset=BASE_FEATURES + TREND_FEATURES).copy()

    df["risk_label"] = df["risk_label"].astype(int)

    print(f"Horizon: {HORIZON} seconds")
    print(f"Rows after fair-comparison filtering: {len(df):,}")
    print(f"Recordings: {df['recording_id'].nunique()}")
    print("Validation: leave one entire recording out")
    print("Probability decision threshold: 0.50")

    all_predictions = []
    fold_metrics = []

    for threshold in THRESHOLDS:
        subset = df[
            df["hypothetical_threshold"] == threshold
        ].copy()

        print("\n" + "-" * 78)
        print(f"HYPOTHETICAL SIGNAL THRESHOLD: {threshold}")
        print(
            f"Rows: {len(subset):,} | "
            f"Positive labels: {subset['risk_label'].sum():,} "
            f"({100 * subset['risk_label'].mean():.2f}%)"
        )

        for model_name in [
            "logistic_regression",
            "random_forest",
        ]:
            for feature_set, features in [
                ("baseline", BASE_FEATURES),
                ("baseline_plus_trends",
                 BASE_FEATURES + TREND_FEATURES),
            ]:
                for held_out in sorted(
                    subset["recording_id"].unique()
                ):
                    train = subset[
                        subset["recording_id"] != held_out
                    ]
                    test = subset[
                        subset["recording_id"] == held_out
                    ]

                    X_train = train[features]
                    y_train = train["risk_label"]
                    X_test = test[features]
                    y_test = test["risk_label"].to_numpy()

                    if y_train.nunique() < 2:
                        print(
                            f"SKIP: threshold={threshold}, "
                            f"model={model_name}, "
                            f"features={feature_set}, "
                            f"held-out={held_out}: "
                            "training labels contain one class"
                        )
                        continue

                    model = make_model(model_name)
                    model.fit(X_train, y_train)

                    positive_index = list(
                        model.classes_
                    ).index(1)

                    probabilities = model.predict_proba(
                        X_test
                    )[:, positive_index]

                    metrics = calculate_metrics(
                        y_test, probabilities
                    )

                    fold_metrics.append({
                        "threshold": threshold,
                        "horizon_sec": HORIZON,
                        "model": model_name,
                        "feature_set": feature_set,
                        "held_out_recording": held_out,
                        **metrics,
                    })

                    all_predictions.append(pd.DataFrame({
                        "threshold": threshold,
                        "horizon_sec": HORIZON,
                        "model": model_name,
                        "feature_set": feature_set,
                        "recording_id": test["recording_id"].to_numpy(),
                        "elapsed_sec": test["elapsed_sec"].to_numpy(),
                        "y_true": y_test,
                        "risk_probability": probabilities,
                    }))

        print(f"Completed threshold {threshold}.")

    if not all_predictions:
        raise RuntimeError("No predictions were generated.")

    predictions_df = pd.concat(
        all_predictions, ignore_index=True
    )
    folds_df = pd.DataFrame(fold_metrics)

    predictions_path = (
        OUTPUT_DIR / "trend_model_oof_predictions.csv"
    )
    folds_path = (
        OUTPUT_DIR / "trend_model_fold_metrics.csv"
    )
    summary_path = (
        OUTPUT_DIR / "trend_model_comparison_summary.csv"
    )

    predictions_df.to_csv(predictions_path, index=False)
    folds_df.to_csv(folds_path, index=False)

    # Pooled out-of-fold metrics. These complement, not replace,
    # the per-recording fold metrics.
    summary_rows = []

    for (threshold, model_name, feature_set), group in (
        predictions_df.groupby(
            ["threshold", "model", "feature_set"]
        )
    ):
        metrics = calculate_metrics(
            group["y_true"].to_numpy(),
            group["risk_probability"].to_numpy(),
        )

        summary_rows.append({
            "threshold": threshold,
            "horizon_sec": HORIZON,
            "model": model_name,
            "feature_set": feature_set,
            **metrics,
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 78)
    print("POOLED OUT-OF-FOLD COMPARISON")
    print("=" * 78)

    columns = [
        "threshold",
        "model",
        "feature_set",
        "f1",
        "recall",
        "precision",
        "average_precision",
        "roc_auc",
    ]

    print(
        summary[columns]
        .sort_values(
            ["threshold", "model", "feature_set"]
        )
        .round(3)
        .to_string(index=False)
    )

    print("\nPER-RECORDING F1 AND RECALL")
    print(
        folds_df[
            [
                "threshold",
                "model",
                "feature_set",
                "held_out_recording",
                "positive_count",
                "f1",
                "recall",
                "precision",
                "roc_auc",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

    print("\nSaved files:")
    print(f"Fold metrics: {folds_path}")
    print(f"OOF predictions: {predictions_path}")
    print(f"Comparison summary: {summary_path}")

    print(
        "\nIMPORTANT: Labels represent hypothetical future "
        "signal-threshold crossings, not verified link failures."
    )


if __name__ == "__main__":
    main()
