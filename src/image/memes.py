"""Hateful Memes loader and the split protocol used for fusion.

Expected layout (the Kaggle mirror and the official release both match):
    <root>/train.jsonl, <root>/dev.jsonl or dev_seen.jsonl, optional test_seen.jsonl, <root>/img/*.png
Each JSON line: {"id": 42953, "img": "img/42953.png", "label": 0/1, "text": "..."}.

Splits (seeded, stratified):
    branch_train  80% of train.jsonl  -> trains the text, image and CLIP branches
    fusion_train  20% of train.jsonl  -> held out from every branch; trains the fusion model
                                         and picks thresholds (out-of-fold)
    test          labelled test_seen.jsonl if present, else dev_seen.jsonl / dev.jsonl
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import SEED

DEFAULT_CANDIDATES = [
    Path("/kaggle/input/facebook-hateful-meme-dataset/data"),
    Path("/kaggle/input/hateful-memes/data"),
    Path("/kaggle/input/hateful-memes"),
    Path("data/raw/hateful_memes"),
]


def find_root(explicit: str | Path | None = None) -> Path:
    """First directory containing train.jsonl: explicit path, MM_MEMES_ROOT, known locations, then /kaggle/input."""
    cands = [Path(explicit)] if explicit else []
    if os.environ.get("MM_MEMES_ROOT"):
        env_root = Path(os.environ["MM_MEMES_ROOT"])
        cands += [env_root] + [p.parent for p in env_root.glob("**/train.jsonl")]
    cands += DEFAULT_CANDIDATES
    cands += [p.parent for p in Path("/kaggle/input").glob("**/train.jsonl")] if Path("/kaggle/input").exists() else []
    for c in cands:
        if (c / "train.jsonl").exists():
            return c
    raise FileNotFoundError("Hateful Memes not found; attach the dataset or pass --memes-root")


def _read_jsonl(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows)


def load_memes(root: str | Path | None = None, fusion_frac: float = 0.2, seed: int = SEED) -> tuple[dict, dict]:
    """Return (splits, meta): the three split frames and a summary for data_summary.json."""
    root = find_root(root)
    train = _read_jsonl(root / "train.jsonl")
    test, test_name = None, None
    # first fully labelled file wins; unlabelled test files are skipped
    for name in ("test_seen.jsonl", "dev_seen.jsonl", "dev.jsonl"):
        p = root / name
        if p.exists():
            df = _read_jsonl(p)
            if "label" in df and df["label"].notna().all():
                test, test_name = df, name
                break
    if test is None:
        raise FileNotFoundError("no labelled evaluation split found")
    for df in (train, test):
        df["img_path"] = df["img"].map(lambda x: str(root / x))
        df["label"] = df["label"].astype(int)
    bt, ft = train_test_split(train, test_size=fusion_frac, random_state=seed, stratify=train["label"])
    splits = {
        "branch_train": bt.reset_index(drop=True),
        "fusion_train": ft.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }
    meta = {
        "root": str(root),
        "test_file": test_name,
        "sizes": {k: {"n": len(v), "hateful": int(v["label"].sum())} for k, v in splits.items()},
    }
    return splits, meta
