"""Tiny synthetic fixtures. Nothing here is real data and nothing here writes to results/."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def synthetic_branch_scores(n_train: int = 400, n_test: int = 300, seed: int = 0) -> dict[str, pd.DataFrame]:
    """Two noisy, partly independent branches so that fusing them should help."""
    rng = np.random.default_rng(seed)
    out = {}
    for split, n in (("fusion_train", n_train), ("test", n_test)):
        y = (rng.random(n) < 0.35).astype(int)
        z_text = 1.2 * y + rng.normal(0, 1, n)
        z_img = 1.0 * y + rng.normal(0, 1, n)
        sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
        ids = np.arange(n) + (0 if split == "fusion_train" else 10_000)
        weapon = rng.random(n) * (0.2 + 0.5 * y)
        out[split] = pd.DataFrame(
            {
                "id": ids,
                "split": split,
                "label": y,
                "p_text": sig(z_text),
                "p_image": sig(z_img),
                "weapon_max_conf": weapon,
                "weapon_present": (weapon > 0.35).astype(int),
                "weapon_count": (weapon > 0.35).astype(int),
                "weapon_class": np.where(weapon > 0.35, "Knife", None),
            }
        )
    return out


@pytest.fixture
def scores_dir(tmp_path):
    s = synthetic_branch_scores()
    both = pd.concat(s.values())
    d = tmp_path / "processed"
    d.mkdir()
    both[["id", "split", "label", "p_text"]].to_csv(d / "meme_scores_text.csv", index=False)
    both[["id", "split", "label", "p_image"]].to_csv(d / "meme_scores_image.csv", index=False)
    both[["id", "split", "label", "weapon_max_conf", "weapon_present", "weapon_count", "weapon_class"]].to_csv(
        d / "meme_scores_weapon.csv", index=False
    )
    return d
