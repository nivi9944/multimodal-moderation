# Results summary

Every number below is read from the listed file in `results/` by `scripts/results_summary.py`.

| Metric | Value | Source |
|---|---|---|
| Jigsaw labelled comments (train + val) | 159,571 | `results/data_summary.json` |
| Jigsaw comments the model was fitted on (train split) | 143,613 | `results/data_summary.json` |
| Jigsaw test comments | 63,978 | `results/data_summary.json` |
| DistilRoBERTa macro-F1 / Harmful F1 / Harmful AUROC | 0.606 / 0.692 / 0.976 | `results/text_eval.json` |
| TF-IDF + LR macro-F1 / Harmful F1 / Harmful AUROC | 0.590 / 0.635 / 0.967 | `results/text_eval.json` |
| VADER macro-F1 / Harmful F1 | 0.416 / 0.264 | `results/text_eval.json` |
| Harmful F1 gain over TF-IDF / VADER (points) | 5.7 / 42.8 | `results/text_eval.json` |
| Macro-F1 gain over TF-IDF (points) | 1.5 | `results/text_eval.json` |
| Toxic-vs-safe AUROC out of domain (youtoxic): DistilRoBERTa / TF-IDF | 0.856 / 0.804 | `results/text_ood.json` |
| Hateful Memes test size | 1,000 | `results/data_summary.json` |
| Hateful Memes AUROC: text only / image only / late fusion | 0.686 / 0.515 / 0.683 | `results/fusion_eval.json` |
| Hateful Memes AUROC: headline model (text_plus_clip) | 0.763 | `results/fusion_eval.json` |
| Harmful recall at t_clear: validation (OOF) / test | 90.1% / 79.0% | `results/threshold.json` |
| Harmful precision at t_clear (test) | 65.1% | `results/threshold.json` |
| Items auto-cleared, skipping manual review (test) | 40.6% | `results/triage.json` |
| API latency, text only: p50 / p95 (ms) | 7.6 / 17.8 | `results/latency.json` |
| API latency, text + image: p50 / p95 (ms) | 262.8 / 324.7 | `results/latency.json` |
| Benchmark CPU threads | 2 | `results/latency.json` |
| ONNX INT8 speed-up vs PyTorch FP32 (p50) / macro-F1 change | 4.88x / +0.0069 | `results/onnx.json` |
| YOLOv8s weapons mAP50 / mAP50-95 (test) | 0.714 / 0.537 | `results/detector_eval.json` |
