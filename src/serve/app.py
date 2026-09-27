"""FastAPI moderation service.

Run:
    uvicorn src.serve.app:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health          which components are loaded
    POST /moderate        multipart form: text, image_url, image (file); any subset, at least one
    POST /moderate/json   JSON body: {"text": ..., "image_url": ..., "image_b64": ...}

Models load once at startup from models/. Missing components degrade gracefully:
text-only requests work with just the TF-IDF baseline; fusion needs the meme text model,
the image model and (optionally) the detector plus models/fusion_lr.joblib.
"""

from __future__ import annotations

import base64
import io
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from src.config import LABELS, MODELS
from src.serve.schemas import Health, ModerateJSON, ModerateResponse, Reason

MAX_IMAGE_BYTES = 5 * 1024 * 1024
TEXT_ONLY_REVIEW = 0.5  # text-only: Harmful argmax but P(Harmful) below this -> Neutral (review)


class Moderator:
    """Holds every loaded component and turns inputs into a decision."""

    def __init__(self, text=None, meme_text=None, image=None, detector=None, fusion: dict | None = None):
        self.text, self.meme_text, self.image, self.detector, self.fusion = text, meme_text, image, detector, fusion

    @classmethod
    def from_disk(cls, models_dir: Path = MODELS) -> Moderator:
        from src.text.infer_text import TransformerScorer, load_text_scorer

        text = load_text_scorer("auto", models_dir)
        meme_text = image = detector = fusion = None
        if (models_dir / "meme_text_distilroberta" / "config.json").exists():
            meme_text = TransformerScorer(models_dir / "meme_text_distilroberta")
        if (models_dir / "effnet_b0_memes.pt").exists():
            from src.image.infer_image import ImageScorer

            image = ImageScorer(models_dir / "effnet_b0_memes.pt")
        if (models_dir / "yolov8s_weapons.pt").exists():
            from src.image.infer_yolo import WeaponDetector

            detector = WeaponDetector(models_dir / "yolov8s_weapons.pt")
        if (models_dir / "fusion_lr.joblib").exists():
            import joblib

            fusion = joblib.load(models_dir / "fusion_lr.joblib")
        return cls(text, meme_text, image, detector, fusion)

    def components(self) -> dict:
        return {
            "text": getattr(self.text, "name", None),
            "meme_text": getattr(self.meme_text, "name", None),
            "image": "efficientnet_b0" if self.image else None,
            "detector": "yolov8s_weapons" if self.detector else None,
            "fusion": ",".join(self.fusion["features"]) if self.fusion else None,
        }

    # ------------------------------------------------------------------ decision
    def moderate(self, text: str | None, image) -> dict:
        if not text and image is None:
            raise ValueError("send text, an image, or both")
        scores: dict[str, float] = {}
        reasons: list[Reason] = []
        det = None
        if text:
            if self.text is None:
                raise RuntimeError("no text model loaded")
            probs = self.text.predict_proba([text])[0]
            for i, lab in enumerate(LABELS):
                scores[f"text_{lab.lower()}"] = round(float(probs[i]), 4)
            scores["p_text_jigsaw"] = float(probs[2])
            if self.meme_text is not None and image is not None:
                scores["p_text"] = float(self.meme_text.predict_proba([text])[0][-1])
        if image is not None:
            if self.image is not None:
                scores["p_image"] = float(self.image.predict([image])[0])
            if self.detector is not None:
                det = self.detector.features([image])[0]
                scores["weapon_max_conf"] = det["weapon_max_conf"]
                scores["weapon_present"] = float(det["weapon_present"])

        if text:
            reasons.append(
                Reason(
                    signal="text_jigsaw",
                    score=round(scores["p_text_jigsaw"], 4),
                    detail="P(Harmful) from the Jigsaw text model",
                )
            )
        if "p_text" in scores:
            reasons.append(Reason(signal="text", score=round(scores["p_text"], 4), detail="meme-caption model"))
        if "p_image" in scores:
            reasons.append(Reason(signal="image", score=round(scores["p_image"], 4), detail="EfficientNet-B0"))
        if det is not None and det["weapon_max_conf"] > 0:
            reasons.append(
                Reason(
                    signal="weapon",
                    score=round(det["weapon_max_conf"], 4),
                    detail=f"{det['weapon_class']} ({'present' if det['weapon_present'] else 'weak'})",
                )
            )

        mode, label, p = self._decide(text, image, scores, det)
        reasons.sort(key=lambda r: r.score, reverse=True)
        if mode == "fusion":
            reasons.insert(0, Reason(signal="fusion", score=round(p, 4), detail="late-fusion P(harmful)"))
        scores["p_harmful"] = p
        return {
            "label": label,
            "p_harmful": round(p, 4),
            "scores": {k: round(v, 4) for k, v in scores.items()},
            "reasons": reasons,
            "mode": mode,
        }

    def _decide(self, text, image, scores, det) -> tuple[str, str, float]:
        if self.fusion is not None and text and image is not None:
            feats = self.fusion["features"]
            needed = {f for f in feats if f != "text_x_image"}
            if needed <= set(scores):
                from src.fusion.calibrate import apply_temperature_binary
                from src.fusion.features import features_from_scores
                from src.fusion.threshold import Thresholds, decide

                x = features_from_scores(scores, feats)
                p_raw = float(self.fusion["model"].predict_proba(x)[0, 1])
                # thresholds act on the raw fusion score; the temperature-scaled value is reported alongside
                label = LABELS[int(decide([p_raw], Thresholds(**self.fusion["thresholds"]))[0])]
                scores["p_harmful_calibrated"] = float(
                    apply_temperature_binary([p_raw], self.fusion.get("temperature", 1.0))[0]
                )
                return "fusion", label, p_raw
        if text:
            probs = [scores[f"text_{lab.lower()}"] for lab in LABELS]
            label = LABELS[int(np.argmax(probs))]
            if label == "Harmful" and probs[2] < TEXT_ONLY_REVIEW:
                label = "Neutral"
            return "text", label, float(probs[2])
        # image only
        p = scores.get("p_image", 0.0)
        label = "Harmful" if p >= 0.5 else "Safe"
        if det is not None and det["weapon_present"] and label == "Safe":
            label = "Neutral"  # weapon seen but image model unsure: send to review
        return "image", label, float(p)


# ---------------------------------------------------------------------- helpers
def _decode_image(raw: bytes):
    from PIL import Image

    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "image larger than 5 MB")
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"could not decode image: {e}") from e


def _fetch(url: str) -> bytes:
    import httpx

    if not url.startswith(("http://", "https://")):
        raise HTTPException(400, "image_url must be http(s)")
    try:
        r = httpx.get(url, timeout=10, follow_redirects=True)
        r.raise_for_status()
    except httpx.HTTPError as e:
        raise HTTPException(400, f"could not fetch image_url: {e}") from e
    return r.content


def create_app(moderator: Moderator | None = None) -> FastAPI:
    state: dict = {"mod": moderator}

    def mod() -> Moderator:
        if state["mod"] is None:
            state["mod"] = Moderator.from_disk()
        return state["mod"]

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        mod()  # load every model once, before the first request
        yield

    app = FastAPI(title="Multimodal Content Moderation", version="1.0.0", lifespan=lifespan)

    def run(text: str | None, image) -> dict:
        t0 = time.perf_counter()
        try:
            out = mod().moderate(text, image)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        except RuntimeError as e:
            raise HTTPException(503, str(e)) from e
        out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return out

    @app.get("/health", response_model=Health)
    def health():
        return {"status": "ok", "components": mod().components()}

    @app.post("/moderate", response_model=ModerateResponse)
    async def moderate(
        text: str | None = Form(None), image_url: str | None = Form(None), image: UploadFile | None = File(None)
    ):
        img = None
        if image is not None:
            img = _decode_image(await image.read())
        elif image_url:
            img = _decode_image(_fetch(image_url))
        return run(text, img)

    @app.post("/moderate/json", response_model=ModerateResponse)
    def moderate_json(body: ModerateJSON):
        img = None
        if body.image_b64:
            img = _decode_image(base64.b64decode(body.image_b64))
        elif body.image_url:
            img = _decode_image(_fetch(body.image_url))
        return run(body.text, img)

    return app


app = create_app()
