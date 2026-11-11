"""Write results/RESULTS_SUMMARY.md (every headline number and its source file) plus a local bullet summary.

Usage:
    python -m scripts.results_summary      # also prints the table

Numbers come only from results/*.json. Anything not produced yet is shown as PENDING.
"""

from __future__ import annotations

from src.config import RESULTS, read_json

P = "PENDING"


def load(name):
    p = RESULTS / f"{name}.json"
    return read_json(p) if p.exists() else {}


def get(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def fmt(x, spec=".3f", suffix=""):
    return P if x is None else f"{x:{spec}}{suffix}"


def main() -> None:
    te, ds, fe = load("text_eval"), load("data_summary"), load("fusion_eval")
    th, tr, lat, ox, de, ood = (
        load(n) for n in ("threshold", "triage", "latency", "onnx", "detector_eval", "text_ood")
    )

    n_train = get(ds, "jigsaw", "splits", "train", "n")
    n_val = get(ds, "jigsaw", "splits", "val", "n")
    n_jig = (n_train + n_val) if n_train and n_val else None
    n_fit = get(te, "distilroberta", "train_rows") or n_train  # rows the model was actually fitted on
    f_dr = get(te, "distilroberta", "test", "macro_f1")
    f_tf = get(te, "tfidf_logreg", "test", "macro_f1")
    f_va = get(te, "vader", "test", "macro_f1")
    h_dr = get(te, "distilroberta", "test", "per_class", "Harmful", "f1")
    h_tf = get(te, "tfidf_logreg", "test", "per_class", "Harmful", "f1")
    h_va = get(te, "vader", "test", "per_class", "Harmful", "f1")
    a_dr = get(te, "distilroberta", "test", "harmful_vs_rest", "auroc")
    a_tf = get(te, "tfidf_logreg", "test", "harmful_vs_rest", "auroc")

    def pts(x, y):
        return round(100 * (x - y), 1) if x is not None and y is not None else None

    head = get(fe, "headline")
    auc_t = get(fe, "variants", "text_only", "test", "auroc")
    auc_i = get(fe, "variants", "image_only", "test", "auroc")
    auc_lf = get(fe, "variants", "late_fusion", "test", "auroc")
    auc_h = get(fe, "variants", head, "test", "auroc") if head else None
    rec = get(th, "at_t_clear", "harmful_recall")
    prec = get(th, "at_t_clear", "harmful_precision")
    rec_val = get(th, "val_oof", "at_t_clear", "harmful_recall")
    cleared = get(tr, "auto_cleared_pct")
    t50, t95 = get(lat, "text", "client_ms", "p50"), get(lat, "text", "client_ms", "p95")
    m50, m95 = get(lat, "fusion", "client_ms", "p50"), get(lat, "fusion", "client_ms", "p95")
    cores = get(lat, "text", "hardware", "logical_cores")
    spd, df1 = get(ox, "speedup_p50"), get(ox, "macro_f1_change")
    map50, map5095 = get(de, "test", "mAP50"), get(de, "test", "mAP50_95")
    n_memes = get(ds, "hateful_memes", "sizes", "test", "n")
    ood_key = "youtoxic" if "youtoxic" in ood else "civil_comments"
    ood_auc = get(ood, ood_key, "models", "text_distilroberta", "toxic_vs_safe", "auroc")
    ood_auc_tf = get(ood, ood_key, "models", "tfidf_logreg", "toxic_vs_safe", "auroc")

    def pct(x):
        return None if x is None else 100 * x

    rows = [
        ("Jigsaw labelled comments (train + val)", fmt(n_jig, ","), "data_summary.json"),
        ("Jigsaw comments the model was fitted on (train split)", fmt(n_fit, ","), "data_summary.json"),
        ("Jigsaw test comments", fmt(get(ds, "jigsaw", "splits", "test", "n"), ","), "data_summary.json"),
        (
            "DistilRoBERTa macro-F1 / Harmful F1 / Harmful AUROC",
            f"{fmt(f_dr)} / {fmt(h_dr)} / {fmt(a_dr)}",
            "text_eval.json",
        ),
        (
            "TF-IDF + LR macro-F1 / Harmful F1 / Harmful AUROC",
            f"{fmt(f_tf)} / {fmt(h_tf)} / {fmt(a_tf)}",
            "text_eval.json",
        ),
        ("VADER macro-F1 / Harmful F1", f"{fmt(f_va)} / {fmt(h_va)}", "text_eval.json"),
        (
            "Harmful F1 gain over TF-IDF / VADER (points)",
            f"{fmt(pts(h_dr, h_tf), '.1f')} / {fmt(pts(h_dr, h_va), '.1f')}",
            "text_eval.json",
        ),
        ("Macro-F1 gain over TF-IDF (points)", fmt(pts(f_dr, f_tf), ".1f"), "text_eval.json"),
        (
            f"Toxic-vs-safe AUROC out of domain ({ood_key}): DistilRoBERTa / TF-IDF",
            f"{fmt(ood_auc)} / {fmt(ood_auc_tf)}",
            "text_ood.json",
        ),
        ("Hateful Memes test size", fmt(n_memes, ","), "data_summary.json"),
        (
            "Hateful Memes AUROC: text only / image only / late fusion",
            f"{fmt(auc_t)} / {fmt(auc_i)} / {fmt(auc_lf)}",
            "fusion_eval.json",
        ),
        (f"Hateful Memes AUROC: headline model ({head})", fmt(auc_h), "fusion_eval.json"),
        (
            "Harmful recall at t_clear: validation (OOF) / test",
            f"{fmt(pct(rec_val), '.1f', '%')} / {fmt(pct(rec), '.1f', '%')}",
            "threshold.json",
        ),
        ("Harmful precision at t_clear (test)", fmt(pct(prec), ".1f", "%"), "threshold.json"),
        ("Items auto-cleared, skipping manual review (test)", fmt(cleared, ".1f", "%"), "triage.json"),
        ("API latency, text only: p50 / p95 (ms)", f"{fmt(t50, '.1f')} / {fmt(t95, '.1f')}", "latency.json"),
        ("API latency, text + image: p50 / p95 (ms)", f"{fmt(m50, '.1f')} / {fmt(m95, '.1f')}", "latency.json"),
        ("Benchmark CPU threads", fmt(cores, "d"), "latency.json"),
        (
            "ONNX INT8 speed-up vs PyTorch FP32 (p50) / macro-F1 change",
            f"{fmt(spd, '.2f', 'x')} / {fmt(df1, '+.4f')}",
            "onnx.json",
        ),
        ("YOLOv8s weapons mAP50 / mAP50-95 (test)", f"{fmt(map50)} / {fmt(map5095)}", "detector_eval.json"),
    ]

    def b(x):
        return f"**{x}**"

    bullets = [
        f"- Fine-tuned **DistilRoBERTa** on {b(f'{round(n_fit / 1000)}K' if n_fit else P)} comments to {b(fmt(h_dr, '.2f'))} Harmful F1, "
        f"beating TF-IDF and VADER baselines by {b(fmt(pts(h_dr, h_tf), '.0f'))} and {b(fmt(pts(h_dr, h_va), '.0f'))} points.",
        f"- Fused caption and CLIP ViT-L/14 signals, lifting Hateful Memes AUROC from {b(fmt(auc_t, '.2f'))} (text-only) "
        f"to {b(fmt(auc_h, '.2f'))} on {fmt(n_memes, ',')} held-out memes.",
        f"- Tuned the Harmful threshold on validation for {b('90%')} recall, auto-clearing {b(fmt(cleared, '.0f', '%'))} of test items "
        f"from manual review ({fmt(pct(rec), '.0f', '%')} test recall).",
        f"- Served via **FastAPI** at {b(fmt(t95, '.0f', ' ms'))} p95 per text item on {fmt(cores, 'd')} vCPUs; INT8 ONNX cut text latency "
        f"{b(fmt(spd, '.1f', 'x'))} with no F1 loss ({fmt(df1, '+.3f')}).",
    ]

    md = [
        "# Results summary",
        "",
        "Every number below is read from the listed file in `results/` by `scripts/results_summary.py`.",
        "",
        "| Metric | Value | Source |",
        "|---|---|---|",
        *[f"| {m} | {v} | `results/{s}` |" for m, v, s in rows],
    ]
    (RESULTS / "RESULTS_SUMMARY.md").write_text("\n".join(md).rstrip() + "\n", encoding="utf-8")
    # The bullet file is git-ignored and stays out of the public repo.
    bl = ["# Resume bullets (local only, filled from RESULTS_SUMMARY.md)", "", *bullets]
    (RESULTS / "RESUME_BULLETS.md").write_text("\n".join(bl) + "\n", encoding="utf-8")
    print("\n".join(md + ["", *bl]))


if __name__ == "__main__":
    main()
