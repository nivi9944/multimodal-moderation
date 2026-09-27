"""API contract tests with stub models (no weights needed)."""

import base64
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sklearn.linear_model import LogisticRegression

from src.serve.app import Moderator, create_app


class StubText:
    name = "stub_text"

    def predict_proba(self, texts, batch=32):
        return np.array([[0.05, 0.05, 0.9] if "hate" in t else [0.9, 0.05, 0.05] for t in texts])


class StubMemeText:
    name = "stub_meme"

    def predict_proba(self, texts, batch=32):
        return np.array([[0.2, 0.8] if "hate" in t else [0.9, 0.1] for t in texts])


class StubImage:
    def predict(self, images, batch=32):
        return np.array([0.7 for _ in images])


class StubDetector:
    def features(self, images, batch=32):
        return [
            {"weapon_max_conf": 0.6, "weapon_present": 1, "weapon_count": 1, "weapon_class": "Knife"} for _ in images
        ]


def fusion_bundle():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    y = (X.sum(1) > 0).astype(int)
    return {
        "model": LogisticRegression().fit(X, y),
        "features": ["p_text", "p_image"],
        "thresholds": {"t_clear": 0.3, "t_flag": 0.8, "target_recall": 0.9, "target_precision": 0.8},
        "temperature": 1.0,
    }


def png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (200, 10, 10)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def client():
    mod = Moderator(
        text=StubText(), meme_text=StubMemeText(), image=StubImage(), detector=StubDetector(), fusion=fusion_bundle()
    )
    return TestClient(create_app(mod))


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["components"]["text"] == "stub_text"


def test_text_only(client):
    r = client.post("/moderate", data={"text": "I hate you"})
    body = r.json()
    assert r.status_code == 200
    assert body["label"] == "Harmful" and body["mode"] == "text"
    assert body["latency_ms"] >= 0 and body["reasons"][0]["signal"] == "text_jigsaw"
    safe = client.post("/moderate", data={"text": "have a nice day"}).json()
    assert safe["label"] == "Safe"


def test_fusion_with_file_upload(client):
    r = client.post("/moderate", data={"text": "hate"}, files={"image": ("a.png", png_bytes(), "image/png")})
    body = r.json()
    assert r.status_code == 200 and body["mode"] == "fusion"
    assert body["reasons"][0]["signal"] == "fusion"
    assert any(x["signal"] == "weapon" and "Knife" in x["detail"] for x in body["reasons"])
    assert set(body["scores"]) >= {"p_text", "p_image", "p_harmful", "weapon_max_conf"}


def test_json_endpoint_image_only(client):
    r = client.post("/moderate/json", json={"image_b64": base64.b64encode(png_bytes()).decode()})
    assert r.status_code == 200 and r.json()["mode"] == "image" and r.json()["label"] == "Harmful"


def test_empty_request_is_422(client):
    assert client.post("/moderate/json", json={}).status_code == 422


def test_bad_image_is_400(client):
    r = client.post("/moderate", files={"image": ("a.png", b"not an image", "image/png")})
    assert r.status_code == 400
