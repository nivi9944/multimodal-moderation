"""Request/response contracts for the moderation API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ModerateJSON(BaseModel):
    text: str | None = Field(None, max_length=10_000)
    image_url: str | None = None
    image_b64: str | None = Field(None, description="base64-encoded image bytes")


class Reason(BaseModel):
    signal: Literal["text", "text_jigsaw", "image", "weapon", "fusion"]
    score: float
    detail: str | None = None


class ModerateResponse(BaseModel):
    label: Literal["Safe", "Neutral", "Harmful"]
    p_harmful: float
    scores: dict[str, float]
    reasons: list[Reason]
    mode: Literal["fusion", "text", "image"]
    latency_ms: float


class Health(BaseModel):
    status: Literal["ok"]
    components: dict[str, str | None]
