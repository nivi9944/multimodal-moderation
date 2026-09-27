"""Optional: export the text model to ONNX, apply dynamic INT8 quantization, and compare on CPU.

Usage:
    python -m src.text.export_onnx [--model-dir models/text_distilroberta] [--n-eval 5000] [--n-latency 300]

Writes models/onnx/text_fp32.onnx, models/onnx/text_int8.onnx and results/onnx.json with
per-item CPU latency (batch 1) for PyTorch FP32 vs ONNX INT8 and macro-F1 of both on the same
stratified Jigsaw test sample.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from src.config import MODELS, RESULTS, SEED, set_seed, write_json
from src.eval.metrics import classification_report
from src.text.data import load_jigsaw
from src.text.infer_text import OnnxScorer, TransformerScorer


def export(model_dir: Path, out: Path) -> tuple[Path, Path]:
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).float().eval()
    enc = tok(["an example sentence"], return_tensors="pt")
    fp32, int8 = out / "text_fp32.onnx", out / "text_int8.onnx"
    kwargs = {}
    if "dynamo" in torch.onnx.export.__code__.co_varnames:
        kwargs["dynamo"] = False  # classic TorchScript exporter: no onnxscript needed, stable for BERT-style models
    torch.onnx.export(
        model,
        (enc["input_ids"], enc["attention_mask"]),
        str(fp32),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"}, "logits": {0: "b"}},
        opset_version=17,
        **kwargs,
    )
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)
    return fp32, int8


def per_item_ms(scorer, texts) -> list[float]:
    for t in texts[:10]:
        scorer.predict_proba([t])
    out = []
    for t in texts:
        t0 = time.perf_counter()
        scorer.predict_proba([t])
        out.append((time.perf_counter() - t0) * 1000)
    return out


def main() -> None:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, default=MODELS / "text_distilroberta")
    ap.add_argument("--n-eval", type=int, default=5000)
    ap.add_argument("--n-latency", type=int, default=300)
    args = ap.parse_args()
    set_seed()
    fp32, int8 = export(args.model_dir, MODELS / "onnx")

    test = load_jigsaw()["test"]
    from sklearn.model_selection import train_test_split

    sample, _ = train_test_split(test, train_size=args.n_eval, random_state=SEED, stratify=test["label"])
    texts, y = sample["text"].tolist(), sample["label"].to_numpy()
    lat_texts = test["text"].sample(args.n_latency, random_state=SEED + 1).tolist()

    pt = TransformerScorer(args.model_dir, device="cpu")
    ox = OnnxScorer(int8, args.model_dir)
    f_pt = classification_report(y, pt.predict_proba(texts, batch=64).argmax(1))["macro_f1"]
    f_ox = classification_report(y, ox.predict_proba(texts, batch=64).argmax(1))["macro_f1"]
    l_pt, l_ox = per_item_ms(pt, lat_texts), per_item_ms(ox, lat_texts)
    res = {
        "model_dir": str(args.model_dir),
        "eval_sample": {"n": len(y), "source": "stratified Jigsaw test sample"},
        "pytorch_fp32": {
            "macro_f1": f_pt,
            "p50_ms": round(float(np.percentile(l_pt, 50)), 2),
            "p95_ms": round(float(np.percentile(l_pt, 95)), 2),
        },
        "onnx_int8": {
            "macro_f1": f_ox,
            "p50_ms": round(float(np.percentile(l_ox, 50)), 2),
            "p95_ms": round(float(np.percentile(l_ox, 95)), 2),
        },
        "speedup_p50": round(float(np.percentile(l_pt, 50) / np.percentile(l_ox, 50)), 2),
        "macro_f1_change": round(f_ox - f_pt, 4),
        "size_mb": {"fp32": round(fp32.stat().st_size / 1e6, 1), "int8": round(int8.stat().st_size / 1e6, 1)},
        "torch_threads": torch.get_num_threads(),
        "batch_size": 1,
    }
    write_json(RESULTS / "onnx.json", res, merge=False)
    print(res)


if __name__ == "__main__":
    main()
