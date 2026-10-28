"""Run with python -m unittest discover -s tests -v."""

import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

from src.baseline import run_baselines
from src.data import load_data


class BaselineTests(unittest.TestCase):
    def test_existing_splits_are_disjoint_and_complete(self):
        X_train, X_cal, X_test, y_train, y_cal, y_test = load_data()
        indices = [set(X.index) for X in (X_train, X_cal, X_test)]
        self.assertFalse(indices[0] & indices[1])
        self.assertFalse(indices[0] & indices[2])
        self.assertFalse(indices[1] & indices[2])
        self.assertEqual(len(set.union(*indices)), 569)
        for X, y in zip((X_train, X_cal, X_test), (y_train, y_cal, y_test)):
            self.assertTrue(X.index.equals(y.index))
            self.assertEqual(set(y), {0, 1})

    def test_training_artifacts_and_training_only_scaling(self):
        X_train, X_cal, X_test, _, y_cal, y_test = load_data()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            metrics = run_baselines(output)
            self.assertEqual(len(metrics), 6)
            pd.testing.assert_frame_equal(metrics, pd.read_csv(output / "metrics.csv"))
            self.assertTrue(metrics[["accuracy", "balanced_accuracy", "roc_auc"]]
                            .apply(lambda values: values.between(0, 1).all()).all())
            for name in metrics.model.unique():
                model = joblib.load(output / f"{name}.joblib")
                if name != "random_forest":
                    scaler = model.steps[0][1]
                    self.assertEqual(scaler.n_samples_seen_, len(X_train))
                    np.testing.assert_allclose(scaler.mean_, X_train.mean())
                else:
                    self.assertEqual(len(model.estimators_samples_[0]), len(X_train))
                for split, X, y in (("calibration", X_cal, y_cal), ("test", X_test, y_test)):
                    saved = pd.read_csv(output / f"{name}_{split}_predictions.csv")
                    np.testing.assert_array_equal(saved.sample_index, X.index)
                    np.testing.assert_array_equal(saved.y_true, y)
                    np.testing.assert_array_equal(saved.y_pred, model.predict(X))
                    probabilities = saved[["probability_0", "probability_1"]].to_numpy()
                    np.testing.assert_allclose(probabilities, model.predict_proba(X))
                    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
                    np.testing.assert_allclose(saved.confidence, probabilities.max(axis=1))
                    report = json.loads((output / f"{name}_{split}_classification_report.json").read_text())
                    self.assertEqual(report["macro avg"]["support"], len(X))


if __name__ == "__main__":
    unittest.main()
