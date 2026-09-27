"""Fail if any tracked text file contains an em dash or en dash (house style)."""

from __future__ import annotations

import sys

from src.config import ROOT

SKIP_DIRS = {".git", ".venv", "venv", "data", "models", "__pycache__", ".ruff_cache", ".pytest_cache"}
EXTS = {".py", ".md", ".ipynb", ".json", ".yml", ".yaml", ".toml", ".txt", ".cfg", ""}
BAD = (chr(0x2014), chr(0x2013))  # em dash, en dash


def main() -> None:
    hits = []
    for p in ROOT.rglob("*"):
        if p.is_dir() or SKIP_DIRS & set(p.relative_to(ROOT).parts) or p.suffix not in EXTS:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if any(b in line for b in BAD):
                hits.append(f"{p.relative_to(ROOT)}:{i}")
    if hits:
        print("dashes found:\n" + "\n".join(hits))
        sys.exit(1)
    print("OK: no em or en dashes")


if __name__ == "__main__":
    main()
