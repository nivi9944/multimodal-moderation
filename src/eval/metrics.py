"""Metric helpers. Every number in results/ is produced through these functions."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from src.config import LABELS


def classification_report(y_true, y_pred, labels=LABELS) -> dict:
    """Accuracy, macro-F1 and per-class precision/recall/F1/support."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    idx = list(range(len(labels)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=idx, zero_division=0)
    return {
        "n": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=idx, average="macro", zero_division=0)), 4),
        "per_class": {
            name: {
                "precision": round(float(p[i]), 4),
                "recall": round(float(r[i]), 4),
                "f1": round(float(f[i]), 4),
                "support": int(s[i]),
            }
            for i, name in enumerate(labels)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=idx).tolist(),
    }


def binary_report(y_true, p_pos, threshold: float = 0.5) -> dict:
    """AUROC, average precision, and F1/macro-F1/accuracy at a probability threshold."""
    y_true, p_pos = np.asarray(y_true).astype(int), np.asarray(p_pos, dtype=float)
    y_hat = (p_pos >= threshold).astype(int)
    out = {
        "n": int(len(y_true)),
        "positive_rate": round(float(y_true.mean()), 4),
        "threshold": round(float(threshold), 4),
        "accuracy": round(float(accuracy_score(y_true, y_hat)), 4),
        "f1_harmful": round(float(f1_score(y_true, y_hat, zero_division=0)), 4),
        "macro_f1": round(float(f1_score(y_true, y_hat, average="macro", zero_division=0)), 4),
    }
    if len(np.unique(y_true)) == 2:
        out["auroc"] = round(float(roc_auc_score(y_true, p_pos)), 4)
        out["average_precision"] = round(float(average_precision_score(y_true, p_pos)), 4)
    return out


def harmful_vs_rest(y3) -> np.ndarray:
    """Collapse 3-class labels to 1 = Harmful, 0 = Safe or Neutral."""
    return (np.asarray(y3) == LABELS.index("Harmful")).astype(int)


def toxic_vs_safe(y3) -> np.ndarray:
    """Collapse 3-class labels to 1 = Neutral or Harmful, 0 = Safe."""
    return (np.asarray(y3) != LABELS.index("Safe")).astype(int)
