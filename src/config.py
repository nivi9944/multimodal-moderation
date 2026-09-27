"""Shared paths, labels, seeds and small I/O helpers."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get("MM_ROOT", Path(__file__).resolve().parents[1]))
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

SEED = 42
LABELS = ["Safe", "Neutral", "Harmful"]
LABEL2ID = {name: i for i, name in enumerate(LABELS)}

# Minimum Harmful recall the triage threshold must reach on validation data.
TARGET_RECALL = 0.90


def set_seed(seed: int = SEED) -> None:
    """Seed python, numpy and (if installed) torch."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _to_builtin(obj: Any) -> Any:
    """Convert numpy scalars/arrays so json.dump works."""
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    return obj


def write_json(path: Path | str, payload: dict, merge: bool = True) -> Path:
    """Write a results file. With merge=True, top-level keys are merged into an existing file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {}
    if merge and path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    data.update(_to_builtin(payload))
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path


def read_json(path: Path | str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
