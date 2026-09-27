"""Load public text datasets and map them to Safe / Neutral / Harmful.

Label mapping for Jigsaw (see DECISIONS.md, D1):

* ``mild`` (default)
    Harmful = severe_toxic | threat | insult | identity_hate | (toxic & obscene)
    Neutral = exactly one of {toxic, obscene} and nothing else ("mild" toxicity)
    Safe    = no label set
* ``spec`` (literal first draft of the plan)
    Harmful = toxic | severe_toxic | threat | insult | identity_hate
    Neutral = obscene only
    Safe    = no label set
  Only ~0.2% of rows are Neutral under ``spec``, so it is kept as an ablation only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DATA_RAW, LABEL2ID, SEED

JIGSAW_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
JIGSAW_URL = (
    "https://raw.githubusercontent.com/nicknochnack/CommentToxicity/main/"
    "jigsaw-toxic-comment-classification-challenge/{name}.csv/{name}.csv"
)


def map_jigsaw(df: pd.DataFrame, scheme: str = "mild") -> np.ndarray:
    """Return integer labels (0 Safe, 1 Neutral, 2 Harmful) for Jigsaw rows."""
    y = np.full(len(df), LABEL2ID["Safe"], dtype=int)
    t, s, o, th, ins, ih = (df[c].to_numpy() == 1 for c in JIGSAW_COLS)
    if scheme == "mild":
        harmful = s | th | ins | ih | (t & o)
        neutral = (t ^ o) & ~harmful
    elif scheme == "spec":
        harmful = t | s | th | ins | ih
        neutral = o & ~harmful
    else:
        raise ValueError(f"unknown scheme {scheme!r}")
    y[neutral] = LABEL2ID["Neutral"]
    y[harmful] = LABEL2ID["Harmful"]
    return y


def _read(name: str, raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / "jigsaw" / f"{name}.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.read_csv(JIGSAW_URL.format(name=name)).to_csv(path, index=False)
    return pd.read_csv(path)


def load_jigsaw(
    scheme: str = "mild", val_size: float = 0.1, raw_dir: Path = DATA_RAW, seed: int = SEED
) -> dict[str, pd.DataFrame]:
    """Kaggle train.csv -> stratified train/val; official test.csv + test_labels.csv -> test.

    Test rows labelled -1 (not scored in the competition) are dropped.
    Each frame has columns: id, text, label.
    """
    train = _read("train", raw_dir)
    test = _read("test", raw_dir).merge(_read("test_labels", raw_dir), on="id")
    test = test[test["toxic"] != -1].reset_index(drop=True)

    frames = {}
    for name, df in (("train_full", train), ("test", test)):
        frames[name] = pd.DataFrame(
            {"id": df["id"], "text": df["comment_text"].astype(str), "label": map_jigsaw(df, scheme)}
        )
    tr, va = train_test_split(
        frames["train_full"], test_size=val_size, random_state=seed, stratify=frames["train_full"]["label"]
    )
    return {
        "train": tr.reset_index(drop=True),
        "val": va.reset_index(drop=True),
        "test": frames["test"],
    }


# ---------------------------------------------------------------- out-of-domain sets
YOUTOXIC_HARM = [
    "IsAbusive",
    "IsThreat",
    "IsHatespeech",
    "IsRacist",
    "IsNationalist",
    "IsSexist",
    "IsHomophobic",
    "IsReligiousHate",
    "IsRadicalism",
]


def map_youtoxic(df: pd.DataFrame) -> np.ndarray:
    """YouToxic English 1000: toxic + any hate/abuse/threat flag -> Harmful,
    toxic with only obscene/provocative flags -> Neutral, not toxic -> Safe."""

    def flag(col: str) -> np.ndarray:
        return df[col].astype(str).str.lower().isin(["true", "1"]).to_numpy() if col in df else np.zeros(len(df), bool)

    toxic = flag("IsToxic")
    harm = np.zeros(len(df), bool)
    for c in YOUTOXIC_HARM:
        harm |= flag(c)
    y = np.full(len(df), LABEL2ID["Safe"], dtype=int)
    y[toxic] = LABEL2ID["Neutral"]
    y[toxic & harm] = LABEL2ID["Harmful"]
    return y


def load_youtoxic(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return pd.DataFrame(
        {"id": df.get("CommentId", df.index), "text": df["Text"].astype(str), "label": map_youtoxic(df)}
    )


def map_civil(df: pd.DataFrame) -> np.ndarray:
    """Civil Comments (fractional annotator scores): toxicity >= 0.5 is toxic;
    toxic and (severe_toxicity | threat | insult | identity_attack) >= 0.5 -> Harmful."""
    tox = df["toxicity"].to_numpy() >= 0.5
    harm = np.zeros(len(df), bool)
    for c in ("severe_toxicity", "threat", "insult", "identity_attack"):
        harm |= df[c].to_numpy() >= 0.5
    y = np.full(len(df), LABEL2ID["Safe"], dtype=int)
    y[tox] = LABEL2ID["Neutral"]
    y[tox & harm] = LABEL2ID["Harmful"]
    return y
