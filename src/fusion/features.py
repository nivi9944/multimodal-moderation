"""Build fusion feature matrices from per-branch score files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED

BRANCH_FILES = ["meme_scores_text.csv", "meme_scores_image.csv", "meme_scores_weapon.csv", "meme_scores_clip.csv"]
# columns in [0, 1] that are logit-transformed before fitting
PROB_COLS = {"p_text", "p_image", "p_text_jigsaw", "p_clip", "weapon_max_conf"}

# Feature sets compared in results/fusion_eval.json. Columns missing from the data are skipped.
VARIANTS: dict[str, list[str]] = {
    "text_only": ["p_text"],
    "image_only": ["p_image"],
    "image_plus_weapon": ["p_image", "weapon_max_conf", "weapon_present"],
    "late_fusion": ["p_text", "p_image", "weapon_max_conf", "weapon_present", "p_text_jigsaw", "text_x_image"],
    "clip_linear_probe": ["p_clip"],
    "text_plus_clip": ["p_text", "p_text_jigsaw", "p_clip"],
    "late_fusion_plus_clip": [
        "p_text",
        "p_image",
        "weapon_max_conf",
        "weapon_present",
        "p_text_jigsaw",
        "text_x_image",
        "p_clip",
    ],
}
DEPLOYED = "late_fusion"


def load_branch_scores(processed: Path = DATA_PROCESSED) -> pd.DataFrame:
    """Inner-join every available branch file on (id, split, label), keeping items scored by all of them."""
    df = None
    for name in BRANCH_FILES:
        p = processed / name
        if not p.exists():
            continue
        part = pd.read_csv(p)
        df = part if df is None else df.merge(part, on=["id", "split", "label"], how="inner")
    if df is None:
        raise FileNotFoundError(f"no meme_scores_*.csv files in {processed}")
    return df


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def make_features(df: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, list[str]]:
    """Return X and the column names actually used. Probabilities are logit-transformed."""
    feats, used = [], []
    for c in cols:
        if c == "text_x_image":
            # interaction term: product of the text and image logits, scaled down by 10
            if {"p_text", "p_image"} <= set(df.columns):
                feats.append(_logit(df["p_text"].to_numpy()) * _logit(df["p_image"].to_numpy()) / 10)
                used.append(c)
            continue
        if c not in df.columns:
            continue
        v = df[c].to_numpy(dtype=float)
        feats.append(_logit(v) if c in PROB_COLS else v)
        used.append(c)
    if not feats:
        return np.zeros((len(df), 0)), []
    return np.column_stack(feats), used


def features_from_scores(scores: dict, cols: list[str]) -> np.ndarray:
    """Single-item version used by the API. `scores` holds the same keys as the columns."""
    return make_features(pd.DataFrame([scores]), cols)[0]
