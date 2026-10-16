
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils import resample
import joblib


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "script25" / "dispatch_training_data.csv"
OUT = ROOT / "data" / "processed" / "script26"
OUT.mkdir(parents=True, exist_ok=True)

TARGET = "dispatch_now"

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

# A cap controls memory and runtime. Sampling is done only on the
# training recordings; each held-out recording is evaluated in full.
MAX_TRAIN_ROWS_PER_RECORDING = 30000
RANDOM_STATE = 42


def metrics_for(y_true, probabilities, threshold=0.5):
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "decision_cutoff": threshold,
        "examples": len(y_true),
        "actual_positive_rate": float(np.mean(y_true)),
        "predicted_positive_rate": float(np.mean(predictions)),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "average_precision": average_precision_score(
            y_true, probabilities
        ),
        "roc_auc": (
            roc_auc_score(y_true, probabilities)
            if len(np.unique(y_true)) > 1 else np.nan
        ),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
    }


def sample_training_recording(group, max_rows):
    if len(group) <= max_rows:
        return group

    # Preserve positive and negative examples where possible.
    parts = []

    for _, class_group in group.groupby(TARGET):
        class_cap = max(1, int(max_rows * len(class_group) / len(group)))
        parts.append(
            resample(
                class_group,
                replace=False,
                n_samples=min(class_cap, len(class_group)),
                random_state=RANDOM_STATE,
            )
        )

    sampled = pd.concat(parts, ignore_index=True)

    if len(sampled) > max_rows:
        sampled = sampled.sample(
            n=max_rows, random_state=RANDOM_STATE
        )

    return sampled


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing training data: {INPUT}")

    df = pd.read_csv(INPUT)

    required = {
        "recording_id", TARGET, "signal_quality_smooth",
        "signal_threshold", *FEATURES
    }
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    for col in FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(
        subset=["recording_id", TARGET, *FEATURES]
    ).copy()

    df[TARGET] = df[TARGET].astype(int)

    recordings = sorted(df["recording_id"].unique())
    if len(recordings) < 3:
        raise ValueError("Need at least three recordings for grouped evaluation.")

    print("=" * 72)
    print("SCRIPT 26 — LEAVE-ONE-RECORDING-OUT DISPATCH MODEL")
    print("=" * 72)
    print(f"Rows: {len(df):,}")
    print(f"Recordings: {recordings}")
    print(f"Features: {len(FEATURES)}")
    print(f"Training cap per recording: {MAX_TRAIN_ROWS_PER_RECORDING:,}")

    prediction_frames = []
    fold_metrics = []

    for fold_number, held_out in enumerate(recordings, start=1):
        train = df[df["recording_id"] != held_out].copy()
        test = df[df["recording_id"] == held_out].copy()

        sampled_parts = []
        for recording_id, group in train.groupby("recording_id"):
            sampled_parts.append(
                sample_training_recording(
                    group, MAX_TRAIN_ROWS_PER_RECORDING
                )
            )

        train_sample = pd.concat(sampled_parts, ignore_index=True)

        model = HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.08,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        )

        model.fit(
            train_sample[FEATURES],
            train_sample[TARGET],
        )

        probabilities = model.predict_proba(test[FEATURES])[:, 1]
        y_true = test[TARGET].to_numpy()

        fold_result = metrics_for(y_true, probabilities)
        fold_result.update({
            "fold": fold_number,
            "held_out_recording": held_out,
            "train_rows_used": len(train_sample),
            "test_rows": len(test),
        })
        fold_metrics.append(fold_result)

        predictions = test[
            [
                "recording_id",
                "elapsed_sec",
                "signal_threshold",
                "relay_speed_mps",
                "setup_time_sec",
                "reconnect_time_sec",
                "signal_quality_smooth",
                TARGET,
            ]
        ].copy()

        predictions["risk_probability"] = probabilities
        predictions["fold"] = fold_number
        prediction_frames.append(predictions)

        print(
            f"\nFold {fold_number}/{len(recordings)} — held out {held_out}"
        )
        print(f"  Training rows used: {len(train_sample):,}")
        print(f"  Test rows: {len(test):,}")
        print(f"  Average precision: {fold_result['average_precision']:.4f}")
        print(f"  F1 at 0.50: {fold_result['f1']:.4f}")
        print(f"  Recall at 0.50: {fold_result['recall']:.4f}")

    oof = pd.concat(prediction_frames, ignore_index=True)
    oof.to_csv(OUT / "dispatch_oof_predictions.csv", index=False)

    # Aggregate all out-of-fold predictions for an overall evaluation.
    overall = metrics_for(
        oof[TARGET].to_numpy(),
        oof["risk_probability"].to_numpy(),
    )
    overall["model"] = "HistGradientBoosting"
    overall["held_out_recording"] = "ALL_OOF"
    overall["train_rows_used"] = np.nan
    overall["test_rows"] = len(oof)

    # Always-negative baseline: useful for interpreting class imbalance.
    always_negative = {
        "model": "AlwaysNegative",
        "held_out_recording": "ALL_OOF",
        "decision_cutoff": np.nan,
        "examples": len(oof),
        "actual_positive_rate": float(oof[TARGET].mean()),
        "predicted_positive_rate": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "average_precision": np.nan,
        "roc_auc": np.nan,
        "true_positive": 0,
        "false_positive": 0,
        "true_negative": int((oof[TARGET] == 0).sum()),
        "false_negative": int((oof[TARGET] == 1).sum()),
    }

    # Reactive baseline: act only once the observed smoothed signal is
    # at or below the scenario threshold. This is a comparator, not a
    # claim that these thresholds represent real radio outages.
    reactive = (
        oof["signal_quality_smooth"] <= oof["signal_threshold"]
    ).astype(int)

    reactive_probabilities = reactive.astype(float)
    reactive_result = metrics_for(
        oof[TARGET].to_numpy(),
        reactive_probabilities,
        threshold=0.5,
    )
    reactive_result["model"] = "ReactiveThreshold"
    reactive_result["held_out_recording"] = "ALL_OOF"
    reactive_result["train_rows_used"] = np.nan
    reactive_result["test_rows"] = len(oof)

    comparison = pd.DataFrame([
        overall,
        always_negative,
        reactive_result,
    ])

    comparison.to_csv(OUT / "model_comparison.csv", index=False)
    pd.DataFrame(fold_metrics).to_csv(
        OUT / "fold_metrics.csv", index=False
    )

    print("\n" + "=" * 72)
    print("OVERALL OUT-OF-FOLD COMPARISON")
    print("=" * 72)
    print(
        comparison[
            ["model", "actual_positive_rate", "predicted_positive_rate",
             "precision", "recall", "f1", "average_precision", "roc_auc"]
        ].round(4).to_string(index=False)
    )

    print("\nSaved:")
    print(OUT / "dispatch_oof_predictions.csv")
    print(OUT / "fold_metrics.csv")
    print(OUT / "model_comparison.csv")

    print(
        "\nNote: this script evaluates generalization to unseen recordings. "
        "It does not yet validate relay placement or real communication recovery."
    )


if __name__ == "__main__":
    main()
