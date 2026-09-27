"""Optional stronger comparison: CLIP (ViT-L/14) image + text embeddings with a linear probe. GPU helps.

Usage:
    python -m src.fusion.clip_probe [--memes-root PATH] [--model openai/clip-vit-large-patch14]
Trains logistic regression on branch_train embeddings; writes data/processed/meme_scores_clip.csv.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV

from src.config import DATA_PROCESSED, RESULTS, SEED, set_seed, write_json
from src.image.memes import load_memes


def embed(df: pd.DataFrame, model, proc, device, batch: int = 64) -> np.ndarray:
    import torch
    from PIL import Image

    out = []
    with torch.inference_mode():
        for i in range(0, len(df), batch):
            chunk = df.iloc[i : i + batch]
            ims = [Image.open(p).convert("RGB") for p in chunk["img_path"]]
            enc = proc(
                text=chunk["text"].astype(str).tolist(),
                images=ims,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            ).to(device)
            o = model(**enc)
            im = torch.nn.functional.normalize(o.image_embeds, dim=-1)
            tx = torch.nn.functional.normalize(o.text_embeds, dim=-1)
            out.append(torch.cat([im, tx, im * tx], -1).float().cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    import torch
    from transformers import CLIPModel, CLIPProcessor

    ap = argparse.ArgumentParser()
    ap.add_argument("--memes-root")
    ap.add_argument("--model", default="openai/clip-vit-large-patch14")
    args = ap.parse_args()
    set_seed()
    splits, _ = load_memes(args.memes_root)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(args.model).to(dev).eval()
    proc = CLIPProcessor.from_pretrained(args.model)
    X = {k: embed(v, model, proc, dev) for k, v in splits.items()}
    clf = LogisticRegressionCV(
        Cs=[0.01, 0.03, 0.1, 0.3, 1, 3],
        cv=5,
        scoring="roc_auc",
        max_iter=3000,
        class_weight="balanced",
        random_state=SEED,
    )
    clf.fit(X["branch_train"], splits["branch_train"]["label"])
    rows = [
        pd.DataFrame(
            {"id": splits[s]["id"], "split": s, "label": splits[s]["label"], "p_clip": clf.predict_proba(X[s])[:, 1]}
        )
        for s in ("fusion_train", "test")
    ]
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    pd.concat(rows).to_csv(DATA_PROCESSED / "meme_scores_clip.csv", index=False)
    write_json(RESULTS / "training_log.json", {"clip_probe": {"model": args.model, "C": float(clf.C_[0])}})


if __name__ == "__main__":
    main()
