"""Fill the README result tables from results/*.json (the only allowed source of numbers).

Usage:
    python -m scripts.render_readme

Replaces everything between <!-- RESULTS:<name> --> and <!-- /RESULTS:<name> --> markers.
A block whose result file does not exist yet shows a "pending" line instead of numbers.
"""

from __future__ import annotations

import re

from src.config import RESULTS, ROOT, read_json

NICE = {
    "majority_class": "Majority class (always Safe)",
    "vader": "VADER compound + thresholds",
    "tfidf_logreg": "TF-IDF + logistic regression",
    "distilroberta": "DistilRoBERTa (fine-tuned)",
    "roberta": "RoBERTa-base (fine-tuned)",
    "text_only": "Text only (DistilRoBERTa on captions)",
    "image_only": "Image only (EfficientNet-B0)",
    "image_plus_weapon": "Image + YOLOv8s weapon features",
    "late_fusion": "Late fusion: text + image + weapon (served by the API)",
    "clip_linear_probe": "CLIP ViT-L/14 linear probe (image + caption)",
    "text_plus_clip": "Text + CLIP fusion",
    "late_fusion_plus_clip": "Late fusion + CLIP",
}


def f(x, pct=False):
    if x is None:
        return "n/a"
    return f"{100 * x:.1f}" if pct else f"{x:.3f}"


def load(name):
    p = RESULTS / f"{name}.json"
    return read_json(p) if p.exists() else None


def pending(what):
    return f"_Pending: {what}._"


def block_text():
    d = load("text_eval")
    if not d:
        return pending("run `make baselines`")
    rows = [
        "| Model | Macro-F1 | Harmful F1 | Neutral F1 | Safe F1 | Harmful AUROC | Accuracy |",
        "|---|---|---|---|---|---|---|",
    ]
    for k in ("majority_class", "vader", "tfidf_logreg", "distilroberta", "roberta"):
        if k in d:
            t = d[k]["test"]
            pc = t["per_class"]
            auc = t.get("harmful_vs_rest", {}).get("auroc")
            rows.append(
                f"| {NICE[k]} | {f(t['macro_f1'])} | {f(pc['Harmful']['f1'])} | {f(pc['Neutral']['f1'])} | "
                f"{f(pc['Safe']['f1'])} | {f(auc)} | {f(t['accuracy'])} |"
            )
    n = d["splits"]["test"]["n"]
    rows.append(f"\nJigsaw official test set, n = {n:,}. Source: `results/text_eval.json`.")
    if "distilroberta" not in d:
        rows.append(pending("DistilRoBERTa row appears after the text notebook"))
    return "\n".join(rows)


def block_ood():
    d = load("text_ood")
    if not d:
        return pending("out-of-domain evaluation (text notebook, step 4)")
    rows = []
    for key in ("youtoxic", "civil_comments"):
        if key not in d:
            continue
        ds = d[key]
        rows += [f"**{ds['dataset']}**", "", "| Model | Macro-F1 | Toxic vs safe AUROC |", "|---|---|---|"]
        for k, v in ds["models"].items():
            name = NICE.get(k.replace("text_", ""), k)
            rows.append(f"| {name} | {f(v['macro_f1'])} | {f(v['toxic_vs_safe'].get('auroc'))} |")
        rows.append("")
    rows.append("Models trained on Jigsaw only, no tuning on these sets. Source: `results/text_ood.json`.")
    return "\n".join(rows)


def block_fusion():
    d = load("fusion_eval")
    if not d:
        return pending("image notebook (Hateful Memes)")
    head = d.get("headline")
    rows = ["| Model | AUROC | Avg. precision | Macro-F1 |", "|---|---|---|---|"]
    for k, v in d["variants"].items():
        t = v["test"]
        name = NICE.get(k, k)
        if k == head:
            name = f"**{name}** (selected on validation)"
        rows.append(f"| {name} | {f(t.get('auroc'))} | {f(t.get('average_precision'))} | {f(t['macro_f1'])} |")
    ds = load("data_summary") or {}
    hm = ds.get("hateful_memes", {})
    split = hm.get("test_file", "see data_summary.json")
    n = hm.get("sizes", {}).get("test", {}).get("n")
    rows.append(
        f"\nHateful Memes `{split}` (n = {n}), never used for training or selection. The headline model is the one with "
        "the best out-of-fold AUROC on the held-out fusion split. Source: `results/fusion_eval.json`."
    )
    return "\n".join(rows)


def block_detector():
    d = load("detector_eval")
    if not d:
        return pending("YOLOv8s training (image notebook, step 3)")
    t = d["test"]
    per = ", ".join(f"{k} {v:.3f}" for k, v in t["per_class_mAP50_95"].items())
    return (
        "| mAP50 | mAP50-95 | Precision | Recall |\n|---|---|---|---|\n"
        f"| {f(t['mAP50'])} | {f(t['mAP50_95'])} | {f(t['precision'])} | {f(t['recall'])} |\n\n"
        f"Per-class mAP50-95: {per}. Source: `results/detector_eval.json`."
    )


def block_triage():
    th, tr = load("threshold"), load("triage")
    if not (th and tr):
        return pending("fusion step")
    rows = [
        "| Model | Harmful recall (val) | Harmful recall (test) | Harmful precision (test) | Auto-cleared (test) |",
        "|---|---|---|---|---|",
    ]
    head = th.get("variant")
    for k, v in th.get("per_variant", {}).items():
        name = NICE.get(k, k)
        cleared = f"{v['test_auto_cleared_pct']:.1f}%"
        if k == head:
            name, cleared = f"**{name}**", f"**{cleared}**"
        rows.append(
            f"| {name} | {f(v['oof_harmful_recall'], True)}% | {f(v['test_harmful_recall'], True)}% | "
            f"{f(v['test_harmful_precision'], True)}% | {cleared} |"
        )
    ref = tr.get("reference_test_oracle_at_target_recall", {}).get("auto_cleared_pct")
    rows.append(
        f"\nFor the headline model, {tr['auto_cleared_pct']:.1f}% of test items fall below t_clear = "
        f"{th['thresholds']['t_clear']:.3f} and would skip review; {tr['harmful_rate_in_cleared_pct']}% of that cleared pile "
        f"is actually hateful (base rate {tr['base_harmful_rate_pct']}%)."
        + (
            f" Setting the threshold on the test labels themselves to force exactly 90% recall would clear {ref:.1f}% "
            "(shown only to size the validation-to-test gap)."
            if ref is not None
            else ""
        )
        + " Sources: `results/threshold.json`, `results/triage.json`."
    )
    return "\n".join(rows)


def block_calibration():
    d = load("calibration")
    if not d:
        return pending("calibration")
    rows = ["| Component | Temperature | Test ECE before | Test ECE after |", "|---|---|---|---|"]
    labels = {
        "tfidf_logreg_text": "TF-IDF text model (3-class)",
        "distilroberta_text": "DistilRoBERTa (3-class)",
        "roberta_text": "RoBERTa-base (3-class)",
    }
    for k, v in d.items():
        name = labels.get(k) or ("Fusion: " + NICE.get(k.replace("fusion_", ""), k) if k.startswith("fusion_") else k)
        rows.append(f"| {name} | {v['temperature']:.2f} | {f(v['test']['ece_before'])} | {f(v['test']['ece_after'])} |")
    rows.append("\nSource: `results/calibration.json`.")
    return "\n".join(rows)


def block_latency():
    d, o = load("latency"), load("onnx")
    if not d and not o:
        return pending("latency benchmark on the laptop (`make serve`, then `make latency`)")
    rows = []
    if d:
        rows += ["| Request type | p50 (ms) | p95 (ms) | Throughput (req/s) | Hardware |", "|---|---|---|---|---|"]
        mode_names = {
            "text": "Text only (ONNX INT8 DistilRoBERTa)",
            "fusion": "Text + image (all branches + late fusion)",
        }
        for k, v in d.items():
            hw = v["hardware"].get("cpu") or v["hardware"].get("machine")
            rows.append(
                f"| {mode_names.get(k, k)} | {v['client_ms']['p50']:.1f} | {v['client_ms']['p95']:.1f} | "
                f"{v['throughput_rps']['value']:.1f} | {hw}, {v['hardware']['logical_cores']} vCPU |"
            )
    if o:
        rows += [
            "",
            "| Text model on CPU (batch 1) | p50 (ms) | Macro-F1 |",
            "|---|---|---|",
            f"| PyTorch FP32 | {o['pytorch_fp32']['p50_ms']:.1f} | {f(o['pytorch_fp32']['macro_f1'])} |",
            f"| ONNX Runtime INT8 | {o['onnx_int8']['p50_ms']:.1f} | {f(o['onnx_int8']['macro_f1'])} |",
            f"\nINT8 speed-up {o['speedup_p50']:.2f}x, macro-F1 change {o['macro_f1_change']:+.4f}. Source: `results/onnx.json`.",
        ]
    return "\n".join(rows)


BLOCKS = {
    "text": block_text,
    "ood": block_ood,
    "fusion": block_fusion,
    "detector": block_detector,
    "triage": block_triage,
    "calibration": block_calibration,
    "latency": block_latency,
}


def main() -> None:
    path = ROOT / "README.md"
    s = path.read_text(encoding="utf-8")
    for name, fn in BLOCKS.items():
        pat = re.compile(rf"(<!-- RESULTS:{name} -->).*?(<!-- /RESULTS:{name} -->)", re.S)
        s = pat.sub(lambda m, fn=fn: m.group(1) + "\n" + fn() + "\n" + m.group(2), s)
    path.write_text(s, encoding="utf-8")
    print("README tables refreshed from results/")


if __name__ == "__main__":
    main()
