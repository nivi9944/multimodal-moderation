# Model card: Multimodal Content Moderation

## Intended use

Research and portfolio prototype for **triaging** user-generated text and images into Safe / Neutral /
Harmful, so that human moderators see fewer items. It is designed to sit in front of human review, never to
replace it: Neutral means "send to a person", and Harmful means "prioritise for a person".

Not intended for: automated account enforcement, legal decisions, languages other than English, or content
types it was not trained on (video, audio, explicit imagery).

## Components

| Component | Training data | Output |
|---|---|---|
| DistilRoBERTa text classifier | Jigsaw Toxic Comments (Wikipedia talk pages) | P(Safe), P(Neutral), P(Harmful) |
| EfficientNet-B0 image classifier | Hateful Memes images | P(hateful) |
| DistilRoBERTa caption classifier | Hateful Memes captions | P(hateful) |
| YOLOv8s weapon detector | Open Images V7: Handgun, Rifle, Shotgun, Knife | boxes, confidence |
| Logistic-regression fusion | Held-out 20% of Hateful Memes train | P(harmful), thresholds |

Metrics for each component are in `results/` and the README.

## Known limitations and biases

* **Identity-term bias.** Toxicity models trained on Jigsaw are known to over-score comments that simply
  mention identity groups (for example "gay", "Muslim", "Black") because those words co-occur with abuse in
  the training data. Check before any real use: score template sentences such as "I am a ___ person" across
  identity terms and compare false-positive rates per group (the Civil Comments identity columns support this).
* **Domain shift.** Wikipedia talk-page language differs from YouTube or social media slang; the out-of-domain
  table in the README quantifies the drop.
* **Label subjectivity.** "Mild" versus "harmful" toxicity is a judgement call; annotator disagreement caps
  achievable Neutral F1.
* **Memes are adversarial by design.** Sarcasm, cultural references and benign confounders defeat all
  branches; see `results/examples.md`.
* **Weapons are not harm by themselves.** Hunting, sport, news and cooking images contain rifles and knives.
  The detector contributes a feature and an explanation, not a verdict; an image-only weapon hit is routed to
  review (Neutral), not auto-flagged.
* **Calibration drifts.** Thresholds and temperatures were fitted on one validation distribution; they must be
  re-fitted when the traffic changes.

## Ethical notes

All data is public and used under its licence. No sexually explicit data is used. Offensive examples appear in
the datasets and in `results/examples.md` only to document failure modes.
