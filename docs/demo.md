# Nov. 13 midterm demo

Run from the repository root using the existing environment:

```sh
.venv/bin/python -m src.demo
```

For a fresh Python 3.13 environment, create a virtual environment and install
`requirements-demo.txt`. This records the numerical-library versions used in
the demo; the broader `requirements.txt` remains unchanged.

The experiment reads the checked-in `data/breast_cancer.csv` and reuses
`src.data.load_data` and the logistic regression pipeline from `src.baseline`.
The stratified seed-42 split is 341/114/114 rows (approximately 60/20/20).
Both the scaler and logistic regression are fitted only on the training rows.
Every table row uses the same 114 test cases and the same fitted model.
The baseline answers all cases. Other rows answer when the largest predicted
class probability is at least the threshold; accuracy uses only answered cases.
An empty answered set has undefined accuracy, displayed as N/A.

Actual stdout is saved in [results/demo.txt](../results/demo.txt).
Additional thresholds 0.90, 0.95, and 0.99 show a wider coverage range than
0.60–0.80. These are descriptive test results, not threshold optimization.
Accuracy need not increase at each threshold; 100% on a smaller answered
subset does not establish a guarantee for future data.

Calibration is a gap: `src/calibration.py` is empty, so no calibrated results
or calibration-fit claim is included. The calibration split stays unused.
`src/evaluation.py` is also empty; the demo uses the focused abstention summary.
Existing saved baseline artifacts are not needed or overwritten by the demo.

Validation:

```sh
.venv/bin/python -m unittest discover -s tests -v
```

All four tests pass. The existing SVM baseline test emits a scikit-learn
deprecation warning for `SVC(probability=True)`; the logistic regression demo
does not use SVM.
