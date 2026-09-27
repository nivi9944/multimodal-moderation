"""Save an fp16 copy of a fine-tuned text classifier (halves the download; loading upcasts to fp32).

Usage:
    python -m scripts.save_half models/text_distilroberta pack/models/text_distilroberta

Runs in its own process so it is unaffected by packages installed later in a notebook session.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


def main() -> None:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    model = AutoModelForSequenceClassification.from_pretrained(src).half()
    model.save_pretrained(dst)
    AutoTokenizer.from_pretrained(src).save_pretrained(dst)
    for extra in src.glob("*.txt"):
        shutil.copy(extra, dst)
    print(f"saved fp16 copy of {src} to {dst}")


if __name__ == "__main__":
    main()
