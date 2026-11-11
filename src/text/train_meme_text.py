"""Text-only Hateful Memes branch: DistilRoBERTa fine-tuned on meme captions (binary). GPU.

Also scores meme captions with the Jigsaw text model (if models/text_distilroberta exists) as an
extra, zero-shot feature p_text_jigsaw.

Usage:
    python -m src.text.train_meme_text [--memes-root PATH] [--epochs 4]
Writes models/meme_text_distilroberta/ and data/processed/meme_scores_text.csv.
"""

from __future__ import annotations

import argparse
import math

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DATA_PROCESSED, MODELS, RESULTS, SEED, set_seed, write_json
from src.image.memes import load_memes


def main() -> None:
    import torch
    from datasets import Dataset
    from scipy.special import softmax
    from sklearn.metrics import roc_auc_score
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--memes-root")
    ap.add_argument("--model", default="distilroberta-base")
    ap.add_argument("--epochs", type=float, default=4)
    args = ap.parse_args()
    set_seed()
    splits, meta = load_memes(args.memes_root)
    tok = AutoTokenizer.from_pretrained(args.model)
    # 10% of branch_train is held out to pick the best epoch by AUROC
    tr, es = train_test_split(
        splits["branch_train"], test_size=0.1, random_state=SEED, stratify=splits["branch_train"]["label"]
    )

    def ds(df):
        d = Dataset.from_dict({"text": df["text"].astype(str).tolist(), "label": df["label"].tolist()})
        return d.map(lambda b: tok(b["text"], truncation=True, max_length=64), batched=True, remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
    out_dir = MODELS / "meme_text_distilroberta"
    targs = TrainingArguments(
        output_dir=str(out_dir / "ckpt"),
        num_train_epochs=args.epochs,
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=128,
        warmup_steps=int(math.ceil(len(tr) / 32) * args.epochs * 0.1),
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        seed=SEED,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="auroc",
        save_total_limit=1,
        report_to=[],
        logging_steps=50,
    )

    def metrics(p):
        return {"auroc": roc_auc_score(p.label_ids, softmax(p.predictions, 1)[:, 1])}

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds(tr),
        eval_dataset=ds(es),
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=metrics,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))

    jig = None
    jdir = MODELS / "text_distilroberta"
    if (jdir / "config.json").exists():
        from src.text.infer_text import TransformerScorer

        jig = TransformerScorer(jdir)

    rows = []
    for split in ("fusion_train", "test"):
        df = splits[split]
        p = softmax(trainer.predict(ds(df)).predictions, 1)[:, 1]
        row = pd.DataFrame({"id": df["id"], "split": split, "label": df["label"], "p_text": p})
        if jig is not None:
            row["p_text_jigsaw"] = jig.predict_proba(df["text"].astype(str).tolist())[:, 2]
        rows.append(row)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    pd.concat(rows).to_csv(DATA_PROCESSED / "meme_scores_text.csv", index=False)
    write_json(
        RESULTS / "training_log.json",
        {
            "meme_text": {
                "log": [h for h in trainer.state.log_history if "eval_auroc" in h],
                "jigsaw_feature": jig is not None,
                "data": meta,
            }
        },
    )
    write_json(RESULTS / "data_summary.json", {"hateful_memes": meta})


if __name__ == "__main__":
    main()
