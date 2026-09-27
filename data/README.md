# Data

Nothing in `data/raw/` is committed. Hashes, sizes and split counts of what was actually used are written to
`results/data_summary.json`.

## 1. Jigsaw Toxic Comment Classification (text train and test)

* Source: Kaggle competition "jigsaw-toxic-comment-classification-challenge" (2018).
* Licence: comment text CC0; underlying Wikipedia talk-page text CC BY-SA 3.0.
* Files: `train.csv` (159,571 rows), `test.csv` + `test_labels.csv` (63,978 scored rows after removing -1).
* Download: automatic. `src/text/data.py` fetches the three CSVs on first use into `data/raw/jigsaw/`
  (from a GitHub mirror of the Kaggle files; the SHA-256 of each file is recorded so it can be checked against
  the Kaggle originals). To use Kaggle directly:
  `kaggle competitions download -c jigsaw-toxic-comment-classification-challenge` and unzip into `data/raw/jigsaw/`.

## 2. YouToxic English 1000 (out-of-domain text test)

* 1,000 YouTube comments with IsToxic, IsHatespeech, IsThreat, IsObscene and related flags.
* Kaggle copy used: `reihanenamdari/youtube-toxicity-data` (CC BY 4.0). Place the CSV in `data/raw/youtoxic/`.
* Mapping: not toxic -> Safe; toxic with a hate, abuse, threat, racism, sexism, homophobia, religious hate,
  nationalism or radicalism flag -> Harmful; other toxic -> Neutral.
* Fallback if unavailable: a seeded 5,000-comment slice of Civil Comments (`google/civil_comments` on Hugging
  Face) via `python -m src.text.eval_ood --civil 5000`. The README states which one was used.

## 3. Hateful Memes (fusion benchmark)

* Source: Meta AI. Accept the licence at the official release before using any copy.
* Kaggle copy used: `williamberrios/hateful-memes` (phase-2 release: `train.jsonl` 8,500, `dev_seen.jsonl`,
  `dev_unseen.jsonl`, labelled `test_seen.jsonl` 1,000, `img/`). The loader evaluates on `test_seen.jsonl` when it
  is labelled, otherwise on the dev split.
* Local path for laptop runs: `data/raw/hateful_memes/`.
* Fallback if access fails: MMHS150K (tweets with images), with the loader adapted and the change stated.

## 4. Open Images V7 weapon subset (detector)

* Classes: Handgun, Rifle, Shotgun, Knife. Downloaded with FiftyOne from the official train / validation / test
  splits (up to 1,000 train images per class, seed 42) by `src/image/train_yolo.py`. Used: 2,600 train, 217
  validation and 571 test images (counts in `results/detector_eval.json`).
* Licence: annotations CC BY 4.0; images CC BY 2.0 as listed by Open Images.
