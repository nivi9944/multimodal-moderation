"""Add the zero-shot Jigsaw-model score (p_text_jigsaw) to the meme caption scores on CPU.

Used when the text and image notebooks run in parallel, so the image notebook had no Jigsaw model.
Needs models/text_distilroberta/ and the meme split files in data/raw/hateful_memes/.

Usage:
    python -m scripts.add_jigsaw_feature
"""

from __future__ import annotations

import pandas as pd

from src.config import DATA_PROCESSED, DATA_RAW, MODELS
from src.image.memes import load_memes
from src.text.infer_text import TransformerScorer


def main() -> None:
    path = DATA_PROCESSED / "meme_scores_text.csv"
    df = pd.read_csv(path)
    if "p_text_jigsaw" in df.columns:
        print("p_text_jigsaw already present; nothing to do")
        return
    splits, _ = load_memes(DATA_RAW / "hateful_memes")
    captions = pd.concat([splits["fusion_train"], splits["test"]])[["id", "text"]].drop_duplicates("id")
    captions["id"] = captions["id"].astype(int)  # jsonl ids may be zero-padded strings; score CSVs store ints
    df["id"] = df["id"].astype(int)
    df = df.merge(captions, on="id", how="left")
    if df["text"].isna().any():
        raise SystemExit("some meme ids have no caption; check data/raw/hateful_memes")
    scorer = TransformerScorer(MODELS / "text_distilroberta", device="cpu")
    df["p_text_jigsaw"] = scorer.predict_proba(df["text"].astype(str).tolist(), batch=64)[:, 2]
    df.drop(columns="text").to_csv(path, index=False)
    print(f"added p_text_jigsaw to {len(df)} rows")


if __name__ == "__main__":
    main()
