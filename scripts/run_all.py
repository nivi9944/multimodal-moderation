"""Reproduce every CPU-side result file in order. GPU stages live in notebooks/.

Usage:
    python -m scripts.run_all                 # data summary, baselines, OOD (if data), fusion (if scores), plots
    python -m scripts.run_all --latency       # also start the API and run the latency benchmark
    python -m scripts.run_all --onnx          # also run the ONNX INT8 comparison (needs the text model)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from src.config import DATA_PROCESSED, DATA_RAW, MODELS


def sh(*args: str) -> None:
    print("\n$", " ".join(args), flush=True)
    subprocess.run([sys.executable, "-m", *args], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latency", action="store_true")
    ap.add_argument("--onnx", action="store_true")
    ap.add_argument("--skip-baselines", action="store_true")
    args = ap.parse_args()

    sh("scripts.data_summary")
    if not args.skip_baselines:
        sh("src.text.baselines")
    yt = sorted((DATA_RAW / "youtoxic").glob("*.csv"))
    if yt:
        sh("src.text.eval_ood", "--youtoxic", str(yt[0]))
    else:
        print("\n[skip] OOD eval: put the YouToxic csv in data/raw/youtoxic/ (or run eval_ood --civil N)")
    if list(DATA_PROCESSED.glob("meme_scores_*.csv")):
        sh("src.fusion.train_fusion")
    else:
        print("\n[skip] fusion: no data/processed/meme_scores_*.csv yet (run the image notebook)")
    if args.onnx and (MODELS / "text_distilroberta" / "config.json").exists():
        sh("src.text.export_onnx")
    if args.latency:
        srv = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.serve.app:app", "--port", "8000"])
        try:
            time.sleep(15)  # give uvicorn time to load the models
            sh("src.eval.latency", "--mode", "text")
            if (MODELS / "fusion_lr.joblib").exists() and Path(DATA_RAW / "hateful_memes").exists():
                sh(
                    "src.eval.latency",
                    "--mode",
                    "fusion",
                    "--n",
                    "300",
                    "--memes-root",
                    str(DATA_RAW / "hateful_memes"),
                )
        finally:
            srv.terminate()
    sh("src.eval.plots")
    sh("scripts.render_readme")
    sh("scripts.results_summary")


if __name__ == "__main__":
    main()
