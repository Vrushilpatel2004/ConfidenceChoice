"""Train baseline classifiers: python -m src.baseline.

Confidence is the largest class probability; ROC-AUC uses P(target=1).
Calibration and test rows are evaluated only, never used for fitting or tuning.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report, roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from .data import load_data
except ImportError:  # Also support python src/baseline.py.
    from data import load_data


DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "results" / "baseline"


def build_models():
    """Return fresh, reproducible estimators with training-only preprocessing."""
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=42)
        ),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "svm": make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)),
    }


def evaluate(model, X, y):
    """Return binary classification metrics, report, and row-level predictions."""
    predicted = model.predict(X)
    probabilities = model.predict_proba(X)
    positive_column = list(model.classes_).index(1)
    metrics = {
        "accuracy": accuracy_score(y, predicted),
        "balanced_accuracy": balanced_accuracy_score(y, predicted),
        "roc_auc": roc_auc_score(y, probabilities[:, positive_column]),
    }
    predictions = pd.DataFrame({
        "sample_index": X.index,
        "y_true": np.asarray(y),
        "y_pred": predicted,
        "confidence": probabilities.max(axis=1),
    })
    for column, label in enumerate(model.classes_):
        predictions[f"probability_{label}"] = probabilities[:, column]
    report = classification_report(y, predicted, output_dict=True, zero_division=0)
    return metrics, report, predictions


def run_baselines(output_dir=DEFAULT_OUTPUT):
    """Fit on the existing training split and persist held-out results/models."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    X_train, X_cal, X_test, y_train, y_cal, y_test = load_data()
    rows = []
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        joblib.dump(model, output_dir / f"{name}.joblib")
        for split, X, y in (
            ("calibration", X_cal, y_cal), ("test", X_test, y_test)
        ):
            metrics, report, predictions = evaluate(model, X, y)
            rows.append({"model": name, "split": split, **metrics})
            predictions.to_csv(output_dir / f"{name}_{split}_predictions.csv", index=False)
            with (output_dir / f"{name}_{split}_classification_report.json").open("w") as f:
                json.dump(report, f, indent=2)
                f.write("\n")
    metrics_frame = pd.DataFrame(rows)
    metrics_frame.to_csv(output_dir / "metrics.csv", index=False)
    return metrics_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(run_baselines(args.output_dir).to_string(index=False))


if __name__ == "__main__":
    main()
