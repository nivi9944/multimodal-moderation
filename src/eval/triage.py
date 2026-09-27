"""Triage metric: how much manual review the model could remove.

Definition (also in the README):
    auto_cleared_pct = share of items with p_harmful < t_clear,
where t_clear is chosen on validation so that Harmful recall >= 90%.
Auto-cleared items are labelled Safe and skip manual review, so auto_cleared_pct is
the estimated reduction in manual-review volume versus reviewing every item.
We always report the cost next to it: the share of truly harmful items that were
auto-cleared (missed) and the harmful rate inside the cleared pile.
"""

from __future__ import annotations

import numpy as np


def triage_report(y, p, t_clear: float) -> dict:
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    cleared = p < t_clear
    n_pos = int((y == 1).sum())
    missed = int((cleared & (y == 1)).sum())
    return {
        "n_items": int(len(y)),
        "t_clear": round(float(t_clear), 6),
        "auto_cleared_pct": round(100 * float(cleared.mean()), 2),
        "sent_to_review_or_flagged_pct": round(100 * float((~cleared).mean()), 2),
        "harmful_recall": round(1 - missed / n_pos, 4) if n_pos else None,
        "harmful_missed": missed,
        "harmful_rate_in_cleared_pct": round(100 * missed / int(cleared.sum()), 2) if cleared.any() else None,
        "base_harmful_rate_pct": round(100 * n_pos / len(y), 2),
    }
