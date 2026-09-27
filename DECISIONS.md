# Design decisions

Short records of the choices that shape the results, with the alternatives considered.

## D1. Jigsaw label mapping: "mild" Neutral class

**Choice.** Harmful = severe_toxic, threat, insult, identity_hate, or (toxic and obscene). Neutral = exactly one of
{toxic, obscene} and nothing else. Safe = no label.

**Why.** The first draft mapped Neutral to "obscene only". In the data that is 317 of 159,571 training comments
(0.2%) and 49 of 63,978 test comments, so a Neutral F1 would be noise and macro-F1 would mostly measure that
noise. The "mild" definition gives a Neutral class of about 3.7% of training comments that matches the product meaning
of Neutral: rude or profane but not attacking anyone. The literal mapping is kept as an ablation
(`ablation_spec_mapping` in `results/text_eval.json`) so the effect is visible.

## D2. Validation split and test set

Kaggle `train.csv` is split 90/10 (stratified, seed 42) into train and validation. The official
`test.csv` + `test_labels.csv` (rows marked -1 removed) is the test set. Every threshold, temperature and
hyper-parameter is chosen on validation; test is touched once per model. Note that the Jigsaw test labels are
known to be harder than the train distribution, so validation scores run higher than test scores for every model.

## D3. Baselines before models

A majority-class row, VADER and TF-IDF + logistic regression are reported on the same splits so the transformer's
gain is measured, not assumed. VADER's two thresholds are grid-searched on validation for macro-F1 to give it a
fair chance.

## D4. DistilRoBERTa over RoBERTa-base

DistilRoBERTa is about 2x faster at inference with a small accuracy cost, fits a free Kaggle GPU with batch 64,
and keeps CPU latency reasonable for serving. RoBERTa-base is an optional notebook flag for the size vs
accuracy comparison.

## D5. Class-weighted loss

Inverse-frequency class weights softened by a square root (normalised to mean 1). Full inverse weights
over-predict Neutral; no weights under-predict it.

## D6. Hateful Memes as the fusion benchmark

It is the standard benchmark where text-only and image-only models both struggle, because meaning comes from
the combination. The official test labels are not public in every mirror, so the loader evaluates on
`test_seen.jsonl` when it is labelled and otherwise on `dev_seen.jsonl` / `dev.jsonl`; the file used is recorded
in `results/data_summary.json`.

## D7. Late fusion with a held-out fusion split

Branches train on 80% of the Hateful Memes train split. The fusion logistic regression trains on the other 20%,
which no branch has seen, so the fusion model learns how much to trust each branch on unseen data instead of
on memorised training scores. Scores enter as logits; a text x image interaction term lets the model reward
agreement. Late fusion was chosen over early fusion (joint encoder) because it is cheap, interpretable (one
weight per signal, which also powers the API "reasons"), and lets each branch be swapped independently. CLIP
(joint image-text embedding) is included as a linear probe to show what an early-fusion style model reaches.

## D8. Two thresholds, recall first

A missed harmful item (false negative) is more costly than an extra review (false positive), so `t_clear` is
the largest threshold that keeps Harmful recall at or above 90% on validation. A second threshold `t_flag`
(precision of at least 0.8 on validation) separates auto-flag Harmful from Neutral (human review). The triage
metric is the share of items below `t_clear`.

## D9. Weapon signal, not NSFW

The internship system this re-implements also screened explicit imagery. This project deliberately uses
weapons and hateful multimodal content as the image harm signal and uses no sexually explicit data.

## D10. Headline model chosen on validation; the API serves the light model

Seven fusion variants are compared. The headline model (text + CLIP) is the one with the best out-of-fold AUROC on
the held-out fusion split, so test labels never influence the choice. The API serves the lighter late-fusion model
(caption models, EfficientNet-B0, YOLOv8s) because CLIP ViT-L/14 adds a large model and several hundred
milliseconds on CPU; serving the CLIP variant is the obvious next step. Its triage numbers are reported separately.

## D11. Reporting the validation-to-test gap instead of hiding it

The 90% recall target is met on validation but not on test (79%): the official test split is more balanced and
includes adversarial "benign confounders". Re-tuning the threshold on test would make the table look better and
the method worse, so the README reports the gap and, for context only, what a test-tuned threshold would clear.

## D12. Reproducibility

Fixed seed 42 everywhere, dataset hashes in `results/data_summary.json`, results written only by code, and
`python -m scripts.run_all` reproduces every CPU-side file. The GPU notebooks are generated from `src/` by
`scripts/build_notebooks.py`, so notebook code and repository code cannot drift apart.
