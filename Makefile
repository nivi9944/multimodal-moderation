PY ?= python

.PHONY: install dev test lint baselines ood fusion plots serve latency onnx notebooks all

install:            ## runtime deps (CPU)
	$(PY) -m pip install -r requirements.txt
dev:                ## light deps for tests/CI
	$(PY) -m pip install -r requirements-dev.txt
test:
	$(PY) -m pytest -q
lint:
	$(PY) -m ruff check src tests scripts
	$(PY) -m scripts.check_dashes
baselines:          ## VADER + TF-IDF on Jigsaw -> results/text_eval.json
	$(PY) -m scripts.data_summary
	$(PY) -m src.text.baselines
ood:                ## needs data/raw/youtoxic/*.csv
	$(PY) -m src.text.eval_ood --youtoxic $(firstword $(wildcard data/raw/youtoxic/*.csv))
fusion:             ## needs data/processed/meme_scores_*.csv from the image notebook
	$(PY) -m src.fusion.train_fusion --memes-root data/raw/hateful_memes
plots:
	$(PY) -m src.eval.plots
serve:
	$(PY) -m uvicorn src.serve.app:app --host 0.0.0.0 --port 8000
latency:            ## start `make serve` in another terminal first
	$(PY) -m src.eval.latency --mode text --n 500
onnx:
	$(PY) -m src.text.export_onnx
notebooks:          ## rebuild notebooks/ from src/
	$(PY) -m scripts.build_notebooks
all:                ## every CPU result file, including latency
	$(PY) -m scripts.run_all --latency --onnx
