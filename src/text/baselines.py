"""Text baselines on Jigsaw: (a) VADER compound score + two thresholds, (b) TF-IDF + logistic regression.

Usage:
    python -m src.text.baselines                # both baselines, default "mild" mapping
    python -m src.text.baselines --skip-ablation

Writes results/text_eval.json (keys: vader, tfidf_logreg, ...), results/calibration.json
(key: tfidf_logreg_text), per-item predictions in data/processed/ and models/tfidf_logreg.joblib.
"""

from __future__ import annotations

import argparse
import time

import joblib
import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED, LABELS, MODELS, RESULTS, set_seed, write_json
from src.eval.metrics import binary_report, classification_report, harmful_vs_rest
from src.fusion.calibrate import ece_multiclass, fit_temperature_logits
from src.text.clean import normalise
from src.text.data import load_jigsaw
from src.text.tfidf import TfidfLogReg


# ------------------------------------------------------------------ VADER
def vader_scores(texts) -> np.ndarray:
    """VADER compound score in [-1, 1] per text, on the first 2000 characters after normalisation."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    sia = SentimentIntensityAnalyzer()
    return np.array([sia.polarity_scores(normalise(t)[:2000])["compound"] for t in texts])


def vader_predict(compound: np.ndarray, t_harm: float, t_neutral: float) -> np.ndarray:
    """compound < t_harm -> Harmful, t_harm <= compound < t_neutral -> Neutral, else Safe."""
    y = np.zeros(len(compound), dtype=int)
    y[compound < t_neutral] = 1
    y[compound < t_harm] = 2
    return y


def fit_vader(compound_val, y_val) -> tuple[float, float, float]:
    """Grid-search the two thresholds (step 0.05, t_harm <= t_neutral) on validation for macro-F1.

    Returns (best macro-F1, t_harm, t_neutral).
    """
    from sklearn.metrics import f1_score

    grid = np.round(np.arange(-0.95, 0.96, 0.05), 2)
    best = (-1.0, 0.0, 0.0)
    for a in grid:
        for b in grid[grid >= a]:
            f = f1_score(y_val, vader_predict(compound_val, a, b), average="macro", labels=[0, 1, 2], zero_division=0)
            if f > best[0]:
                best = (float(f), float(a), float(b))
    return best


def _save_preds(name: str, split: str, df: pd.DataFrame, probs: np.ndarray | None, pred: np.ndarray, extra=None):
    """Write per-item predictions to data/processed/text_preds_<name>_<split>.csv."""
    out = pd.DataFrame({"id": df["id"], "label": df["label"], "pred": pred})
    if probs is not None:
        for i, lab in enumerate(LABELS):
            out[f"p_{lab.lower()}"] = probs[:, i]
    if extra is not None:
        for k, v in extra.items():
            out[k] = v
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out.to_csv(DATA_PROCESSED / f"text_preds_{name}_{split}.csv", index=False)


def run_vader(data) -> dict:
    t0 = time.time()
    c_val, c_test = vader_scores(data["val"]["text"]), vader_scores(data["test"]["text"])
    f_val, a, b = fit_vader(c_val, data["val"]["label"].to_numpy())
    pred = vader_predict(c_test, a, b)
    _save_preds("vader", "test", data["test"], None, pred, {"compound": c_test})
    rep = classification_report(data["test"]["label"], pred)
    # negated compound, so a higher score means more harmful
    rep["harmful_vs_rest"] = binary_report(harmful_vs_rest(data["test"]["label"]), -c_test, threshold=-a)
    return {
        "thresholds": {"harmful_below": a, "neutral_below": b},
        "val_macro_f1": round(f_val, 4),
        "test": rep,
        "seconds": round(time.time() - t0, 1),
    }


def run_majority(data) -> dict:
    """Always predict the most frequent training class (reference floor for macro-F1)."""
    maj = int(np.bincount(data["train"]["label"]).argmax())
    return {
        "predicts": LABELS[maj],
        "test": classification_report(data["test"]["label"], np.full(len(data["test"]), maj)),
    }


def run_tfidf(data, save_model: bool = True) -> tuple[dict, dict]:
    """Fit TF-IDF + LR on train. Returns (evaluation report, calibration report)."""
    t0 = time.time()
    model = TfidfLogReg().fit(data["train"]["text"].tolist(), data["train"]["label"].to_numpy())
    fit_s = time.time() - t0
    p_val = model.predict_proba(data["val"]["text"].tolist())
    p_test = model.predict_proba(data["test"]["text"].tolist())
    pred = p_test.argmax(1)
    rep = classification_report(data["test"]["label"], pred)
    rep["harmful_vs_rest"] = binary_report(harmful_vs_rest(data["test"]["label"]), p_test[:, 2])
    out = {
        "val": classification_report(data["val"]["label"], p_val.argmax(1)),
        "test": rep,
        "train_seconds": round(fit_s, 1),
    }
    if save_model:
        MODELS.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODELS / "tfidf_logreg.joblib", compress=3)
        _save_preds("tfidf", "val", data["val"], p_val, p_val.argmax(1))
        _save_preds("tfidf", "test", data["test"], p_test, pred)
    # calibration: temperature on log-probabilities (valid logits up to a constant)
    lv, lt = np.log(np.clip(p_val, 1e-9, 1)), np.log(np.clip(p_test, 1e-9, 1))
    t = fit_temperature_logits(data["val"]["label"].to_numpy(), lv)
    from scipy.special import softmax

    cal = {
        "temperature": round(t, 4),
        "val": {
            "ece_before": round(ece_multiclass(data["val"]["label"], p_val), 4),
            "ece_after": round(ece_multiclass(data["val"]["label"], softmax(lv / t, 1)), 4),
        },
        "test": {
            "ece_before": round(ece_multiclass(data["test"]["label"], p_test), 4),
            "ece_after": round(ece_multiclass(data["test"]["label"], softmax(lt / t, 1)), 4),
        },
        "note": "Top-label ECE, 15 bins, temperature fitted on validation.",
    }
    return out, cal


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", default="mild", choices=["mild", "spec"])
    ap.add_argument("--skip-ablation", action="store_true")
    args = ap.parse_args()
    set_seed()

    data = load_jigsaw(args.scheme)
    sizes = {
        k: {"n": len(v), "label_counts": {LABELS[i]: int((v["label"] == i).sum()) for i in range(3)}}
        for k, v in data.items()
    }
    print("splits", sizes)

    majority = run_majority(data)
    vader = run_vader(data)
    print("VADER test macro-F1", vader["test"]["macro_f1"])
    tfidf, cal = run_tfidf(data)
    print("TF-IDF test macro-F1", tfidf["test"]["macro_f1"])

    payload = {
        "dataset": "Jigsaw Toxic Comment Classification (Kaggle train.csv; official test.csv + test_labels.csv, rows with -1 dropped)",
        "label_scheme": args.scheme,
        "splits": sizes,
        "majority_class": majority,
        "vader": vader,
        "tfidf_logreg": tfidf,
    }
    if not args.skip_ablation:
        other = "spec" if args.scheme == "mild" else "mild"
        d2 = load_jigsaw(other)
        abl, _ = run_tfidf(d2, save_model=False)
        payload[f"ablation_{other}_mapping"] = {
            "splits": {k: {LABELS[i]: int((v["label"] == i).sum()) for i in range(3)} for k, v in d2.items()},
            "tfidf_logreg_test": abl["test"],
        }
        print(f"TF-IDF ({other} mapping) test macro-F1", abl["test"]["macro_f1"])

    write_json(RESULTS / "text_eval.json", payload)
    write_json(RESULTS / "calibration.json", {"tfidf_logreg_text": cal})


if __name__ == "__main__":
    main()
