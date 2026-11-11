"""Out-of-domain text evaluation: models trained on Jigsaw, tested on YouTube comments.

Usage:
    python -m src.text.eval_ood --youtoxic path/to/youtoxic_english_1000.csv
    python -m src.text.eval_ood --civil 5000          # fallback: Civil Comments test slice (HF datasets)

Every text model found in models/ (TF-IDF, transformers) plus VADER (with the thresholds
learned on Jigsaw validation) is scored. Writes results/text_ood.json.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import MODELS, RESULTS, SEED, read_json, write_json
from src.eval.metrics import binary_report, classification_report, toxic_vs_safe
from src.text.data import load_youtoxic, map_civil
from src.text.infer_text import TfidfScorer, TransformerScorer


def load_civil(n: int) -> pd.DataFrame:
    """Seeded random slice of n rows from the Civil Comments test split."""
    from datasets import load_dataset

    ds = load_dataset("google/civil_comments", split="test").shuffle(seed=SEED).select(range(n))
    df = ds.to_pandas()
    return pd.DataFrame({"id": np.arange(len(df)), "text": df["text"].astype(str), "label": map_civil(df)})


def evaluate(df: pd.DataFrame) -> dict:
    """3-class report per model, plus toxic-vs-safe AUROC with 1 - P(Safe) as the score."""
    out: dict = {}
    y = df["label"].to_numpy()
    texts = df["text"].tolist()
    scorers = []
    if (MODELS / "tfidf_logreg.joblib").exists():
        scorers.append(TfidfScorer(MODELS / "tfidf_logreg.joblib"))
    for d in sorted(MODELS.glob("text_*")):
        if (d / "config.json").exists():
            scorers.append(TransformerScorer(d))
    for s in scorers:
        p = s.predict_proba(texts)
        rep = classification_report(y, p.argmax(1))
        rep["toxic_vs_safe"] = binary_report(toxic_vs_safe(y), 1 - p[:, 0])
        out[s.name] = rep
    te = RESULTS / "text_eval.json"
    if te.exists() and "vader" in read_json(te):
        from src.text.baselines import vader_predict, vader_scores

        th = read_json(te)["vader"]["thresholds"]
        c = vader_scores(texts)
        rep = classification_report(y, vader_predict(c, th["harmful_below"], th["neutral_below"]))
        # negated compound, cut at the Neutral threshold learned on Jigsaw
        rep["toxic_vs_safe"] = binary_report(toxic_vs_safe(y), -c, threshold=-th["neutral_below"])
        out["vader"] = rep
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--youtoxic", type=Path)
    ap.add_argument("--civil", type=int, default=0)
    args = ap.parse_args()
    if args.youtoxic:
        df, name = load_youtoxic(args.youtoxic), "YouToxic English 1000"
    elif args.civil:
        df, name = load_civil(args.civil), f"Civil Comments test slice (n={args.civil}, seed {SEED})"
    else:
        raise SystemExit("pass --youtoxic PATH or --civil N")
    counts = {lab: int((df["label"] == i).sum()) for i, lab in enumerate(["Safe", "Neutral", "Harmful"])}
    res = evaluate(df)
    key = "youtoxic" if args.youtoxic else "civil_comments"
    write_json(
        RESULTS / "text_ood.json",
        {
            key: {
                "dataset": name,
                "label_counts": counts,
                "models": res,
                "note": "Models trained on Jigsaw only; no tuning on this set.",
            }
        },
    )
    for k, v in res.items():
        print(f"{k:>16}: macro-F1 {v['macro_f1']}  toxic-vs-safe AUROC {v['toxic_vs_safe'].get('auroc')}")


if __name__ == "__main__":
    main()
