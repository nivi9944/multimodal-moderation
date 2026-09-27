"""Text normalisation shared by the baselines, the transformer and the API."""

from __future__ import annotations

import html
import re

_URL = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_USER = re.compile(r"@\w+")
_IP = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_WIKI_META = re.compile(r"\(talk\)|\(UTC\)|\bUTC\b", re.IGNORECASE)
_REPEAT = re.compile(r"(.)\1{3,}")  # "soooooo" -> "sooo"
_NON_TEXT = re.compile(r"[^a-z0-9'!?.,\s*]")
_SPACE = re.compile(r"\s+")


def normalise(text: str | None) -> str:
    """Light cleaning that keeps casing and punctuation (used for the transformer and VADER).

    VADER uses capitalisation and "!" as intensity cues, and subword models handle casing,
    so we only strip markup, links, user handles, IPs and Wikipedia signature noise.
    """
    if text is None:
        return ""
    text = html.unescape(str(text))
    text = _URL.sub(" ", text)
    text = _USER.sub(" ", text)
    text = _IP.sub(" ", text)
    text = _WIKI_META.sub(" ", text)
    text = _REPEAT.sub(r"\1\1\1", text)
    return _SPACE.sub(" ", text).strip()


def clean_for_tfidf(text: str | None) -> str:
    """Aggressive cleaning for bag-of-words models: lowercase and drop symbols.

    '*' is kept on purpose so masked profanity such as "f*ck" stays one token.
    """
    text = normalise(text).lower()
    text = _NON_TEXT.sub(" ", text)
    return _SPACE.sub(" ", text).strip()
