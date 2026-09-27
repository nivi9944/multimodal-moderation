"""Text scorer used by the API and the OOD evaluation.

Loads the fine-tuned transformer from models/text_<name>/ if present (optionally an
ONNX INT8 export), otherwise falls back to the TF-IDF + LR baseline in
models/tfidf_logreg.joblib. Every scorer returns probabilities over LABELS.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config import MODELS
from src.text.clean import normalise


class TransformerScorer:
    backend = "transformer"

    def __init__(self, model_dir: Path, max_len: int = 128, device: str | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).float().to(self.device).eval()
        self.max_len = max_len
        self.name = Path(model_dir).name

    def predict_proba(self, texts: list[str], batch: int = 32) -> np.ndarray:
        out = []
        with self.torch.inference_mode():
            for i in range(0, len(texts), batch):
                enc = self.tok(
                    [normalise(t) for t in texts[i : i + batch]],
                    truncation=True,
                    max_length=self.max_len,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
                out.append(self.torch.softmax(self.model(**enc).logits, -1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 3))


class OnnxScorer:
    backend = "onnx"

    def __init__(self, onnx_path: Path, tok_dir: Path, max_len: int = 128):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self.sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.tok = AutoTokenizer.from_pretrained(tok_dir)
        self.max_len = max_len
        self.name = Path(onnx_path).stem
        self.inputs = {i.name for i in self.sess.get_inputs()}

    def predict_proba(self, texts: list[str], batch: int = 32) -> np.ndarray:
        from scipy.special import softmax

        out = []
        for i in range(0, len(texts), batch):
            enc = self.tok(
                [normalise(t) for t in texts[i : i + batch]],
                truncation=True,
                max_length=self.max_len,
                padding=True,
                return_tensors="np",
            )
            feed = {k: v.astype(np.int64) for k, v in enc.items() if k in self.inputs}
            out.append(softmax(self.sess.run(None, feed)[0], -1))
        return np.concatenate(out) if out else np.zeros((0, 3))


class TfidfScorer:
    backend = "tfidf"
    name = "tfidf_logreg"

    def __init__(self, path: Path):
        import joblib

        self.model = joblib.load(path)

    def predict_proba(self, texts: list[str], batch: int = 0) -> np.ndarray:
        return self.model.predict_proba(list(texts))


def load_text_scorer(prefer: str = "auto", models_dir: Path = MODELS):
    """prefer: auto | onnx | transformer | tfidf."""
    onnx = models_dir / "onnx" / "text_int8.onnx"
    tdirs = sorted(p for p in models_dir.glob("text_*") if (p / "config.json").exists())
    if prefer in ("auto", "onnx") and onnx.exists() and tdirs:
        return OnnxScorer(onnx, tdirs[0])
    if prefer in ("auto", "transformer") and tdirs:
        pick = next((p for p in tdirs if "distilroberta" in p.name), tdirs[0])
        return TransformerScorer(pick)
    if (models_dir / "tfidf_logreg.joblib").exists():
        return TfidfScorer(models_dir / "tfidf_logreg.joblib")
    return None
