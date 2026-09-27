"""Record dataset versions, licences, sizes and splits in results/data_summary.json (Jigsaw part).

The GPU notebooks add the hateful_memes and open_images_weapons entries; eval_ood adds the OOD set.
"""

from __future__ import annotations

import hashlib

from src.config import DATA_RAW, LABELS, RESULTS, SEED, write_json
from src.text.data import JIGSAW_URL, load_jigsaw


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    data = load_jigsaw("mild")
    files = {n: sha256(DATA_RAW / "jigsaw" / f"{n}.csv") for n in ("train", "test", "test_labels")}
    write_json(
        RESULTS / "data_summary.json",
        {
            "jigsaw": {
                "name": "Jigsaw Toxic Comment Classification Challenge (Kaggle, 2018)",
                "licence": "Comment text CC0; underlying Wikipedia talk-page text CC BY-SA 3.0",
                "download": JIGSAW_URL.format(name="<train|test|test_labels>")
                + " (byte-identical mirror of the Kaggle files; see sha256)",
                "sha256": files,
                "label_mapping": "mild (see src/text/data.py and DECISIONS.md D1)",
                "split_protocol": f"train.csv -> 90/10 stratified train/val (seed {SEED}); test.csv + test_labels.csv (rows != -1) -> test",
                "splits": {
                    k: {"n": len(v), **{LABELS[i]: int((v["label"] == i).sum()) for i in range(3)}}
                    for k, v in data.items()
                },
            }
        },
    )


if __name__ == "__main__":
    main()
