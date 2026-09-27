import numpy as np
import pytest

from src.eval.metrics import binary_report, classification_report, harmful_vs_rest, toxic_vs_safe
from src.eval.triage import triage_report
from src.fusion.calibrate import (
    apply_temperature_binary,
    ece_binary,
    ece_multiclass,
    fit_temperature_binary,
    fit_temperature_logits,
)
from src.fusion.threshold import decide, fit_thresholds, recall_precision_at, threshold_for_recall


def test_classification_report_perfect_and_macro():
    r = classification_report([0, 1, 2, 2], [0, 1, 2, 2])
    assert r["accuracy"] == 1 and r["macro_f1"] == 1
    r = classification_report([0, 0, 1, 2], [0, 0, 0, 0])
    assert r["per_class"]["Harmful"]["recall"] == 0
    assert r["confusion_matrix"][2] == [1, 0, 0]


def test_binary_report_auroc():
    r = binary_report([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert r["auroc"] == 1.0 and r["f1_harmful"] == 1.0


def test_label_collapses():
    assert harmful_vs_rest([0, 1, 2]).tolist() == [0, 0, 1]
    assert toxic_vs_safe([0, 1, 2]).tolist() == [0, 1, 1]


@pytest.mark.parametrize("target", [0.5, 0.8, 0.9, 1.0])
def test_threshold_for_recall_meets_target_and_is_tight(target):
    rng = np.random.default_rng(1)
    y = (rng.random(500) < 0.3).astype(int)
    p = np.clip(0.5 * y + rng.normal(0.3, 0.2, 500), 0, 1)
    t = threshold_for_recall(y, p, target)
    r, _, _ = recall_precision_at(y, p, t)
    assert r >= target
    # any slightly higher threshold must fall below the target (largest valid threshold)
    higher = p[p > t]
    if len(higher):
        r2, _, _ = recall_precision_at(y, p, higher.min())
        assert r2 < target


def test_decide_three_way():
    th = fit_thresholds(np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1]), np.linspace(0, 1, 10), 0.9, 0.9)
    assert th.t_clear <= th.t_flag
    d = decide([0.0, (th.t_clear + th.t_flag) / 2, 1.0], th)
    assert d[0] == 0 and d[-1] == 2


def test_triage_counts():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.9, 0.05, 0.8])
    r = triage_report(y, p, t_clear=0.3)
    assert r["auto_cleared_pct"] == 60.0
    assert r["harmful_missed"] == 1 and r["harmful_recall"] == 0.5
    assert r["harmful_rate_in_cleared_pct"] == pytest.approx(33.33, abs=0.01)


def test_ece_and_temperature():
    rng = np.random.default_rng(0)
    p_true = rng.random(20000)
    y = (rng.random(20000) < p_true).astype(int)
    assert ece_binary(y, p_true) < 0.02
    over = apply_temperature_binary(p_true, 0.4)  # overconfident version
    assert ece_binary(y, over) > 0.05
    t = fit_temperature_binary(y, over)
    assert 2.2 < t < 2.8  # undoes the 1/0.4 sharpening
    assert ece_binary(y, apply_temperature_binary(over, t)) < 0.02


def test_multiclass_temperature():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 3000)
    logits = rng.normal(0, 1, (3000, 3))
    logits[np.arange(3000), y] += 1.0
    t = fit_temperature_logits(y, logits * 3)
    assert t > 1.5
    probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    assert 0 <= ece_multiclass(y, probs) < 1
