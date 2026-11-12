"""Selective prediction using the largest predicted class probability."""

import numpy as np


def summarize_abstention(y_true, y_pred, probabilities, threshold):
    """Answer when max(predict_proba) >= threshold; empty accuracy is None."""
    answered_mask = np.asarray(probabilities).max(axis=1) >= threshold
    answered = int(answered_mask.sum())
    total = len(answered_mask)
    return {
        "answered": answered,
        "skipped": total - answered,
        "coverage": answered / total if total else 0.0,
        "accuracy": float(np.mean(
            np.asarray(y_true)[answered_mask] == np.asarray(y_pred)[answered_mask]
        )) if answered else None,
    }
