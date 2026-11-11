"""Weapon detector features for fusion and for the API explanation.

Usage (scores every meme in fusion_train and test):
    python -m src.image.infer_yolo [--memes-root PATH]
Writes data/processed/meme_scores_weapon.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED, MODELS

CONF_PRESENT = 0.35  # a box above this confidence counts as "weapon present"


class WeaponDetector:
    def __init__(self, weights: Path = MODELS / "yolov8s_weapons.pt", device: str | None = None):
        from ultralytics import YOLO, settings

        settings.update({"sync": False})  # no usage analytics
        self.model = YOLO(str(weights))
        self.device = device

    def features(self, images: list, batch: int = 32) -> list[dict]:
        """images: PIL images or paths. Returns one feature dict per image."""
        feats = []
        for i in range(0, len(images), batch):
            # low conf floor so weapon_max_conf still carries signal below CONF_PRESENT
            for r in self.model.predict(images[i : i + batch], conf=0.05, verbose=False, device=self.device):
                conf = r.boxes.conf.cpu().numpy() if len(r.boxes) else np.zeros(0)
                cls = r.boxes.cls.cpu().numpy().astype(int) if len(r.boxes) else np.zeros(0, int)
                top = int(conf.argmax()) if len(conf) else -1
                feats.append(
                    {
                        "weapon_max_conf": float(conf.max()) if len(conf) else 0.0,
                        "weapon_present": int(bool(len(conf)) and conf.max() >= CONF_PRESENT),
                        "weapon_count": int((conf >= CONF_PRESENT).sum()),
                        "weapon_class": r.names[int(cls[top])] if top >= 0 else None,
                    }
                )
        return feats


def main() -> None:
    from src.image.memes import load_memes

    ap = argparse.ArgumentParser()
    ap.add_argument("--memes-root")
    args = ap.parse_args()
    splits, _ = load_memes(args.memes_root)
    det = WeaponDetector()
    rows = []
    for split in ("fusion_train", "test"):
        df = splits[split]
        f = pd.DataFrame(det.features(df["img_path"].tolist()))
        rows.append(pd.concat([pd.DataFrame({"id": df["id"], "split": split, "label": df["label"]}), f], axis=1))
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    pd.concat(rows).to_csv(DATA_PROCESSED / "meme_scores_weapon.csv", index=False)


if __name__ == "__main__":
    main()
