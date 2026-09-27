"""Fine-tune EfficientNet-B0 (ImageNet-pretrained, timm) as the image-only Hateful Memes branch. GPU.

Usage:
    python -m src.image.train_effnet [--memes-root PATH] [--epochs 8]

Trains on branch_train (with a 10% early-stopping slice), then scores fusion_train and test.
Writes models/effnet_b0_memes.pt and data/processed/meme_scores_image.csv.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.config import DATA_PROCESSED, MODELS, RESULTS, SEED, set_seed, write_json
from src.image.infer_image import build_model, build_transforms
from src.image.memes import load_memes


def main() -> None:
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    ap = argparse.ArgumentParser()
    ap.add_argument("--memes-root")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--no-pretrained", action="store_true", help="random init (offline smoke tests only)")
    args = ap.parse_args()
    set_seed()
    splits, meta = load_memes(args.memes_root)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    class DS(Dataset):
        def __init__(self, df, train):
            self.df, self.tf = df.reset_index(drop=True), build_transforms(train)

        def __len__(self):
            return len(self.df)

        def __getitem__(self, i):
            r = self.df.iloc[i]
            return self.tf(Image.open(r["img_path"]).convert("RGB")), torch.tensor(float(r["label"]))

    tr, es = train_test_split(
        splits["branch_train"], test_size=0.1, random_state=SEED, stratify=splits["branch_train"]["label"]
    )
    dl = DataLoader(DS(tr, True), batch_size=args.batch, shuffle=True, num_workers=2, drop_last=True)

    def loader(df):
        return DataLoader(DS(df, False), batch_size=args.batch * 2, num_workers=2)

    model = build_model(pretrained=not args.no_pretrained).to(dev)
    pos_weight = torch.tensor((tr["label"] == 0).sum() / max((tr["label"] == 1).sum(), 1), device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(dl))
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")

    def predict(df) -> np.ndarray:
        model.eval()
        out = []
        with torch.inference_mode():
            for x, _ in loader(df):
                with torch.autocast(dev, enabled=dev == "cuda"):
                    out.append(torch.sigmoid(model(x.to(dev)).squeeze(-1)).float().cpu().numpy())
        return np.concatenate(out)

    best, best_state, log = -1.0, None, []
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            with torch.autocast(dev, enabled=dev == "cuda"):
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    model(x).squeeze(-1), y, pos_weight=pos_weight
                )
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
        auc = roc_auc_score(es["label"], predict(es))
        log.append({"epoch": ep + 1, "early_stop_auroc": round(float(auc), 4)})
        print(log[-1])
        if auc > best:
            best, best_state = auc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, MODELS / "effnet_b0_memes.pt")

    rows = []
    for split in ("fusion_train", "test"):
        df = splits[split]
        rows.append(pd.DataFrame({"id": df["id"], "split": split, "label": df["label"], "p_image": predict(df)}))
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    pd.concat(rows).to_csv(DATA_PROCESSED / "meme_scores_image.csv", index=False)
    write_json(
        RESULTS / "training_log.json",
        {
            "effnet_b0": {
                "epochs": log,
                "best_early_stop_auroc": round(best, 4),
                "train_seconds": round(time.time() - t0, 1),
                "data": meta,
            }
        },
    )


if __name__ == "__main__":
    main()
