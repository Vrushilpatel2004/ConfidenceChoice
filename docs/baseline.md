# Baseline models

From the repository root, install `requirements.txt` and run:

```sh
python -m src.baseline
python -m unittest discover -s tests -v
```

Use `--output-dir PATH` to override `results/baseline/`.

The existing CSV contains 569 samples, 30 numeric features, no missing
values, and target counts of 212 for class 0 and 357 for class 1.
The existing stratified split in `src/data.py` supplies 341 training,
114 calibration, and 114 test samples, with random seed 42.

Logistic Regression and SVM use StandardScaler pipelines. Random Forest
uses 200 trees. All models use random seed 42; only training rows are
passed to fit, including fitting the scalers. SVM uses probability=True,
whose internal probability estimation uses training data only. No external
calibration or test rows are used for fitting or hyperparameter selection.

Each model is saved as a joblib file (including its scaler when applicable).
For each held-out split, prediction CSVs contain the original zero-based
CSV row index, true label, predicted label, confidence (maximum class
probability), and probabilities for both classes. SVM's predicted label
can differ from the class with the largest estimated probability.
ROC-AUC uses the probability of target 1. Metrics are saved in metrics.csv;
classification reports are saved as JSON, including per-class precision,
recall, F1, and support. These are baseline probability estimates; the
external calibration split remains available for subsequent calibration.

Tests use the standard-library unittest runner and temporary output folders.
They check split separation, scaler statistics, artifact round trips,
probability consistency, and classification report support.

The installed scikit-learn 1.9.1 warns that SVC(probability=True) will be
removed in 1.11. The requested setting is retained here.
