"""Unpack results_*.zip files from the GPU notebooks into the repo.

Usage:
    python -m scripts.merge_results results_text.zip results_image.zip

JSON files under results/ are merged key by key (so CPU-side keys such as the Jigsaw
ablation are kept); every other file (weights, score CSVs, plots) is written in place.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

from src.config import ROOT


def merge(zip_path: Path) -> list[str]:
    done = []
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            dest = ROOT / info.filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            raw = z.read(info)
            if dest.suffix == ".json" and dest.parent.name == "results" and dest.exists():
                current = json.loads(dest.read_text(encoding="utf-8"))
                current.update(json.loads(raw))
                dest.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
            else:
                dest.write_bytes(raw)
            done.append(info.filename)
    return done


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for p in sys.argv[1:]:
        files = merge(Path(p))
        print(f"{p}: {len(files)} files")


if __name__ == "__main__":
    main()
