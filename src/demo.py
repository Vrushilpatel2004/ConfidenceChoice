"""Reproduce the Nov. 13 midterm experiment: python -m src.demo."""

import sklearn

from .abstention import summarize_abstention
from .baseline import build_models
from .data import load_data


def main():
    X_train, X_cal, X_test, y_train, _, y_test = load_data()
    model = build_models()["logistic_regression"]
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)
    predictions = model.predict(X_test)

    print("ConfidenceChoice | Breast Cancer Wisconsin (Diagnostic)")
    print(f"scikit-learn={sklearn.__version__}; split seed=42; model seed=42")
    print(f"Split: train={len(X_train)}, calibration={len(X_cal)}, test={len(X_test)}")
    print("StandardScaler + LogisticRegression(max_iter=2000); fit on train only")
    print("Test rule: answer if max(predict_proba) >= threshold")
    print(f"{'threshold':>10} {'answered':>9} {'skipped':>8} {'coverage':>10} {'accuracy':>10}")
    # Include high-confidence thresholds to make the coverage tradeoff visible.
    for threshold in (0.0, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99):
        result = summarize_abstention(y_test, predictions, probabilities, threshold)
        label = "baseline" if threshold == 0 else f"{threshold:.2f}"
        accuracy = "N/A" if result["accuracy"] is None else f'{result["accuracy"]:.2%}'
        print(f'{label:>10} {result["answered"]:>9} {result["skipped"]:>8} '
              f'{result["coverage"]:>10.2%} {accuracy:>10}')
    print("Accuracy is among answered cases; baseline answers every test case.")
    print("Calibration: not implemented (src/calibration.py is empty); calibration split unused.")
    print("Thresholds illustrate test-set behavior; no threshold selected or tuned on test.")


if __name__ == "__main__":
    main()
