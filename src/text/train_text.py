"""Fine-tune DistilRoBERTa (or RoBERTa) on Jigsaw with a class-weighted loss. GPU recommended.

Usage (Kaggle / Colab):
    python -m src.text.train_text --model distilroberta-base --epochs 2
    python -m src.text.train_text --model roberta-base --epochs 2      # optional size comparison

Writes models/text_<short>/, per-item predictions in data/processed/, and merges
<short> into results/text_eval.json and <short>_text into results/calibration.json.
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED, LABELS, MODELS, RESULTS, SEED, set_seed, write_json
from src.eval.metrics import binary_report, classification_report, harmful_vs_rest
from src.fusion.calibrate import ece_multiclass, fit_temperature_logits
from src.text.clean import normalise
from src.text.data import load_jigsaw


def short_name(model: str) -> str:
    return model.split("/")[-1].replace("-base", "").replace("-", "_")


def class_weights(y: np.ndarray, power: float = 0.5) -> np.ndarray:
    """Inverse-frequency weights softened by `power` (0.5 = sqrt) and normalised to mean 1."""
    counts = np.bincount(y, minlength=len(LABELS)).astype(float)
    w = (counts.sum() / np.maximum(counts, 1)) ** power
    return w / w.mean()


def main() -> None:
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilroberta-base")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--scheme", default="mild")
    ap.add_argument("--limit", type=int, default=0, help="subsample train for smoke tests")
    args = ap.parse_args()
    set_seed()
    name = short_name(args.model)

    data = load_jigsaw(args.scheme)
    if args.limit:
        data["train"] = data["train"].sample(args.limit, random_state=SEED)
    tok = AutoTokenizer.from_pretrained(args.model)

    def to_ds(df: pd.DataFrame) -> Dataset:
        ds = Dataset.from_dict({"text": [normalise(t) for t in df["text"]], "label": df["label"].tolist()})
        return ds.map(
            lambda b: tok(b["text"], truncation=True, max_length=args.max_len), batched=True, remove_columns=["text"]
        )

    ds = {k: to_ds(v) for k, v in data.items()}
    weights = torch.tensor(class_weights(data["train"]["label"].to_numpy()), dtype=torch.float)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=weights.to(outputs.logits.device))
            return (loss, outputs) if return_outputs else loss

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id={lab: i for i, lab in enumerate(LABELS)},
    )
    out_dir = MODELS / f"text_{name}"
    targs = TrainingArguments(
        output_dir=str(out_dir / "ckpt"),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch * 2,
        warmup_steps=int(math.ceil(len(ds["train"]) / args.batch) * args.epochs * 0.06),
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        seed=SEED,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=200,
        report_to=[],
    )
    trainer = WeightedTrainer(
        model=model,
        args=targs,
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
        data_collator=DataCollatorWithPadding(tok),
    )
    t0 = time.time()
    trainer.train()
    train_s = time.time() - t0

    logits = {k: trainer.predict(ds[k]).predictions for k in ("val", "test")}
    from scipy.special import softmax

    probs = {k: softmax(v, 1) for k, v in logits.items()}
    for split in ("val", "test"):
        df = pd.DataFrame({"id": data[split]["id"], "label": data[split]["label"], "pred": probs[split].argmax(1)})
        for i, lab in enumerate(LABELS):
            df[f"p_{lab.lower()}"] = probs[split][:, i]
            df[f"logit_{lab.lower()}"] = logits[split][:, i]
        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_PROCESSED / f"text_preds_{name}_{split}.csv", index=False)

    rep = classification_report(data["test"]["label"], probs["test"].argmax(1))
    rep["harmful_vs_rest"] = binary_report(harmful_vs_rest(data["test"]["label"]), probs["test"][:, 2])
    write_json(
        RESULTS / "text_eval.json",
        {
            name: {
                "base_model": args.model,
                "epochs": args.epochs,
                "max_len": args.max_len,
                "lr": args.lr,
                "batch": args.batch,
                "class_weights": weights.tolist(),
                "train_rows": len(data["train"]),
                "val": classification_report(data["val"]["label"], probs["val"].argmax(1)),
                "test": rep,
                "train_seconds": round(train_s, 1),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            }
        },
    )

    t = fit_temperature_logits(data["val"]["label"].to_numpy(), logits["val"])
    write_json(
        RESULTS / "calibration.json",
        {
            f"{name}_text": {
                "temperature": round(t, 4),
                "val": {
                    "ece_before": round(ece_multiclass(data["val"]["label"], probs["val"]), 4),
                    "ece_after": round(ece_multiclass(data["val"]["label"], softmax(logits["val"] / t, 1)), 4),
                },
                "test": {
                    "ece_before": round(ece_multiclass(data["test"]["label"], probs["test"]), 4),
                    "ece_after": round(ece_multiclass(data["test"]["label"], softmax(logits["test"] / t, 1)), 4),
                },
                "note": "Top-label ECE, 15 bins, temperature fitted on validation logits.",
            }
        },
    )

    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    (out_dir / "temperature.txt").write_text(f"{t:.6f}\n")
    print(f"{name}: test macro-F1 {rep['macro_f1']}  saved to {out_dir}")


if __name__ == "__main__":
    main()
