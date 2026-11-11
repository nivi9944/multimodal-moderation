"""Figures for the README. All plots read from results/ or data/processed/ only.

Usage:
    python -m src.eval.plots      # regenerate every plot whose inputs exist
"""

from __future__ import annotations

import json

import matplotlib

# headless backend; it has to be selected before pyplot is imported
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from src.config import LABELS, RESULTS, read_json  # noqa: E402

PALETTE = {
    "text_only": "#4C78A8",
    "image_only": "#F58518",
    "image_plus_weapon": "#E45756",
    "late_fusion": "#54A24B",
    "clip_linear_probe": "#B279A2",
    "late_fusion_plus_clip": "#9D755D",
}
# the last model present in text_eval.json is the deployed one, and that is the one plotted
TEXT_MODEL_ORDER = ["vader", "tfidf_logreg", "roberta", "distilroberta"]


def plot_confusion_text() -> str | None:
    """Row-normalised Jigsaw test confusion matrix. Returns the PNG path, or None if there is nothing to plot."""
    p = RESULTS / "text_eval.json"
    if not p.exists():
        return None
    ev = read_json(p)
    name = next((m for m in reversed(TEXT_MODEL_ORDER) if m in ev), None)
    if name is None:
        return None
    cm = np.array(ev[name]["test"]["confusion_matrix"], dtype=float)
    norm = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(
                j,
                i,
                f"{norm[i, j]:.2f}\n({int(cm[i, j])})",
                ha="center",
                va="center",
                color="white" if norm[i, j] > 0.5 else "black",
                fontsize=8,
            )
    ax.set_xticks(range(3), LABELS)
    ax.set_yticks(range(3), LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Jigsaw test, {name} (row-normalised)", fontsize=10)
    fig.tight_layout()
    out = RESULTS / "confusion_text.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)


def plot_fusion(curves: dict, out_dir=None) -> None:
    """AUROC bar chart and Harmful precision-recall curves. curves: {variant: (y_true, p)} on the test split."""
    out_dir = out_dir or RESULTS
    from sklearn.metrics import roc_auc_score

    fig, ax = plt.subplots(figsize=(6, 3.6))
    names = list(curves)
    aucs = [roc_auc_score(*curves[n]) for n in names]
    bars = ax.barh(names, aucs, color=[PALETTE.get(n, "#999") for n in names])
    for b, a in zip(bars, aucs):
        ax.text(a + 0.005, b.get_y() + b.get_height() / 2, f"{a:.3f}", va="center", fontsize=8)
    ax.axvline(0.5, color="#888", lw=0.8, ls="--")
    ax.set_xlim(0.45, max(aucs) + 0.06)
    ax.set_xlabel("AUROC (Hateful Memes test)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_dir / "fusion_vs_single.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    for n, (y, p) in curves.items():
        pr, rc, _ = precision_recall_curve(y, p)
        ax.plot(rc, pr, label=n, color=PALETTE.get(n), lw=2 if n == "late_fusion" else 1.2)
    ax.axvline(0.9, color="#888", lw=0.8, ls="--")
    ax.text(0.905, 0.95, "90% recall", fontsize=7, color="#555")
    ax.set_xlabel("Harmful recall")
    ax.set_ylabel("Harmful precision")
    ax.set_title("Precision-recall, Hateful Memes test", fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "pr_curve_harmful.png", dpi=150)
    plt.close(fig)


def plot_latency(samples: dict[str, list[float]]) -> None:
    """samples: {mode: [ms, ...]} from results/latency_samples.json, one panel per mode."""
    modes = [m for m in ("text", "fusion") if m in samples]
    fig, axes = plt.subplots(1, len(modes), figsize=(5 * len(modes), 3.2), squeeze=False)
    titles = {"text": "text only (ONNX INT8)", "fusion": "text + image (late fusion)"}
    for ax, mode in zip(axes[0], modes):
        s = np.asarray(samples[mode])
        ax.hist(s, bins=40, color="#4C78A8")
        for q, ls in ((50, "-"), (95, "--")):
            v = np.percentile(s, q)
            ax.axvline(v, color="#E45756", ls=ls, lw=1)
            ax.text(v, ax.get_ylim()[1] * 0.9, f" p{q} {v:.0f} ms", fontsize=7, color="#E45756")
        ax.set_xlabel("Latency per request (ms)")
        ax.set_ylabel("Requests")
        ax.set_title(f"/moderate, {titles.get(mode, mode)}, CPU", fontsize=10)
    fig.tight_layout()
    fig.savefig(RESULTS / "latency_hist.png", dpi=150)
    plt.close(fig)


def main() -> None:
    made = [plot_confusion_text()]
    lat = RESULTS / "latency_samples.json"
    if lat.exists():
        d = json.loads(lat.read_text())
        plot_latency({k: v for k, v in d.items() if isinstance(v, list)})
        made.append("latency_hist.png")
    print("plots:", [m for m in made if m])


if __name__ == "__main__":
    main()
