"""Expected calibration error and temperature scaling."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, log_softmax, logit


def ece_binary(y, p, n_bins: int = 15) -> float:
    """ECE for P(Harmful): bin by predicted probability, average |accuracy - confidence|."""
    y, p = np.asarray(y).astype(int), np.clip(np.asarray(p, float), 0, 1)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def ece_multiclass(y, probs, n_bins: int = 15) -> float:
    """Top-label ECE for multiclass probabilities (n, k)."""
    y, probs = np.asarray(y), np.asarray(probs, float)
    conf = probs.max(1)
    correct = (probs.argmax(1) == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, bins[1:-1]), 0, n_bins - 1)
    return float(
        sum(
            (idx == b).mean() * abs(correct[idx == b].mean() - conf[idx == b].mean())
            for b in range(n_bins)
            if (idx == b).any()
        )
    )


def fit_temperature_binary(y, p) -> float:
    """Temperature T minimising NLL of sigmoid(logit(p) / T)."""
    y = np.asarray(y).astype(float)
    z = logit(np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6))

    def nll(t):
        q = np.clip(expit(z / t), 1e-9, 1 - 1e-9)
        return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))

    return float(minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded").x)


def apply_temperature_binary(p, t: float) -> np.ndarray:
    z = logit(np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6))
    return expit(z / t)


def fit_temperature_logits(y, logits) -> float:
    """Temperature for multiclass logits (n, k)."""
    y, logits = np.asarray(y), np.asarray(logits, float)

    def nll(t):
        return -np.mean(log_softmax(logits / t, axis=1)[np.arange(len(y)), y])

    return float(minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded").x)


def calibration_report(y_val, p_val, y_test, p_test) -> dict:
    """Fit T on validation, report ECE before/after on validation and test."""
    t = fit_temperature_binary(y_val, p_val)
    return {
        "temperature": round(t, 4),
        "val": {
            "ece_before": round(ece_binary(y_val, p_val), 4),
            "ece_after": round(ece_binary(y_val, apply_temperature_binary(p_val, t)), 4),
        },
        "test": {
            "ece_before": round(ece_binary(y_test, p_test), 4),
            "ece_after": round(ece_binary(y_test, apply_temperature_binary(p_test, t)), 4),
        },
    }
