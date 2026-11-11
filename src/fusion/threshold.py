"""Threshold selection for the Harmful score.

Two thresholds turn one probability p_harmful into three actions:

    p <  t_clear            -> Safe     (auto-cleared, no human review)
    t_clear <= p < t_flag   -> Neutral  (sent to manual review)
    p >= t_flag             -> Harmful  (auto-flagged, high precision)

t_clear is the largest threshold that keeps Harmful recall >= TARGET_RECALL on validation.
t_flag is the smallest threshold that reaches the target precision on validation.
Both are chosen on validation and applied unchanged to test.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from src.config import TARGET_RECALL


@dataclass
class Thresholds:
    """The two cut-offs on p_harmful and the targets they were fitted for."""

    t_clear: float
    t_flag: float
    target_recall: float
    target_precision: float

    def to_dict(self) -> dict:
        return {k: round(float(v), 6) for k, v in asdict(self).items()}


def recall_precision_at(y, p, t: float) -> tuple[float, float, float]:
    """Harmful recall, precision and flagged share when flagging p >= t."""
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    flagged = p >= t
    tp = int((flagged & (y == 1)).sum())
    pos = int((y == 1).sum())
    recall = tp / pos if pos else float("nan")
    precision = tp / int(flagged.sum()) if flagged.any() else float("nan")
    return recall, precision, float(flagged.mean())


def threshold_for_recall(y, p, target: float = TARGET_RECALL) -> float:
    """Largest threshold t such that recall(p >= t) >= target.

    With positive scores sorted in descending order, flagging everything >= the k-th one
    gives recall k / pos, so k = ceil(target * pos).
    """
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    pos_scores = np.sort(p[y == 1])[::-1]
    if len(pos_scores) == 0:
        raise ValueError("no positive examples to set a recall threshold")
    k = int(np.ceil(target * len(pos_scores)))
    k = min(max(k, 1), len(pos_scores))
    return float(pos_scores[k - 1])


def threshold_for_precision(y, p, target: float = 0.8, floor: float = 0.5) -> float:
    """Smallest threshold with precision >= target over at least 5 flagged items.

    Falls back to max(floor, highest score) when no threshold qualifies.
    """
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    candidates = np.unique(p)
    for t in candidates:  # ascending
        _, prec, _ = recall_precision_at(y, p, t)
        if not np.isnan(prec) and prec >= target and (p >= t).sum() >= 5:
            return float(t)
    return float(max(floor, candidates.max()))


def fit_thresholds(y_val, p_val, target_recall: float = TARGET_RECALL, target_precision: float = 0.8) -> Thresholds:
    """Fit t_clear for the recall target and t_flag for the precision target; t_flag is never below t_clear."""
    t_clear = threshold_for_recall(y_val, p_val, target_recall)
    t_flag = max(threshold_for_precision(y_val, p_val, target_precision), t_clear)
    return Thresholds(t_clear, t_flag, target_recall, target_precision)


def decide(p, th: Thresholds) -> np.ndarray:
    """Vectorised 3-way decision: 0 Safe, 1 Neutral (review), 2 Harmful."""
    p = np.asarray(p, float)
    out = np.ones(len(p), dtype=int)
    out[p < th.t_clear] = 0
    out[p >= th.t_flag] = 2
    return out


def threshold_report(y, p, th: Thresholds) -> dict:
    """Harmful recall, precision and flagged share at t_clear and at t_flag."""
    r, pr, flagged = recall_precision_at(y, p, th.t_clear)
    rf, prf, flagged_f = recall_precision_at(y, p, th.t_flag)
    return {
        "thresholds": th.to_dict(),
        "at_t_clear": {
            "harmful_recall": round(r, 4),
            "harmful_precision": round(pr, 4),
            "flagged_share": round(flagged, 4),
        },
        "at_t_flag": {
            "harmful_recall": round(rf, 4),
            "harmful_precision": round(prf, 4),
            "flagged_share": round(flagged_f, 4),
        },
    }
