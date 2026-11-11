"""Fine-tune YOLOv8s on an Open Images V7 weapon subset (Handgun, Rifle, Shotgun, Knife). GPU.

Usage:
    python -m src.image.train_yolo [--per-class 1000] [--epochs 40]

1. Downloads the subset with FiftyOne (train / validation / test splits of Open Images V7),
   keeping only boxes of the four classes, and exports it in YOLO format.
2. Trains yolov8s.pt (COCO-pretrained) with a fixed seed.
3. Evaluates on the held-out test split and writes results/detector_eval.json
   (mAP50, mAP50-95, precision, recall, per-class mAP50-95).
Weights: models/yolov8s_weapons.pt
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

from src.config import DATA_RAW, MODELS, RESULTS, SEED, set_seed, write_json

CLASSES = ["Handgun", "Rifle", "Shotgun", "Knife"]


def build_dataset(out_dir: Path, per_class: int) -> tuple[Path, dict]:
    import fiftyone as fo
    import fiftyone.zoo as foz
    from fiftyone import ViewField as F

    counts = {}
    # validation and test get a fifth of the train budget per class, at least 50
    budget = {"train": per_class, "validation": max(per_class // 5, 50), "test": max(per_class // 5, 50)}
    for split, yolo_split in (("train", "train"), ("validation", "val"), ("test", "test")):
        views = []
        for cls in CLASSES:  # per-class download keeps the classes balanced
            ds = foz.load_zoo_dataset(
                "open-images-v7",
                split=split,
                label_types=["detections"],
                classes=[cls],
                max_samples=budget[split],
                seed=SEED,
                shuffle=True,
                dataset_name=f"weapons-{split}-{cls}",
            )
            views.append(ds)
        merged = fo.Dataset(f"weapons-{split}-merged", overwrite=True)
        for v in views:
            merged.add_samples(v.filter_labels("ground_truth", F("label").is_in(CLASSES)))
        merged = merged.filter_labels("ground_truth", F("label").is_in(CLASSES))
        counts[yolo_split] = {"images": len(merged), "boxes": merged.count("ground_truth.detections")}
        merged.export(
            export_dir=str(out_dir),
            dataset_type=fo.types.YOLOv5Dataset,
            label_field="ground_truth",
            split=yolo_split,
            classes=CLASSES,
        )
    return out_dir / "dataset.yaml", counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=1000, help="train images per class before filtering")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--weights", default="yolov8s.pt", help="starting checkpoint (COCO-pretrained yolov8s by default)")
    ap.add_argument("--data-yaml", type=Path, help="skip download and use an existing YOLO dataset.yaml")
    args = ap.parse_args()
    set_seed()
    from ultralytics import YOLO, settings

    settings.update({"sync": False})  # no usage analytics

    if args.data_yaml:
        yaml_path, counts = args.data_yaml, {"note": "pre-built dataset"}
    else:
        yaml_path, counts = build_dataset(DATA_RAW / "weapons_yolo", args.per_class)

    t0 = time.time()
    model = YOLO(args.weights)
    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=SEED,
        deterministic=True,
        patience=15,
        project=str(MODELS / "yolo_runs"),
        name="weapons",
        exist_ok=True,
    )
    train_s = time.time() - t0
    best = Path(model.trainer.best)
    MODELS.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, MODELS / "yolov8s_weapons.pt")

    m = YOLO(str(MODELS / "yolov8s_weapons.pt")).val(
        data=str(yaml_path),
        split="test",
        imgsz=args.imgsz,
        seed=SEED,
        project=str(MODELS / "yolo_runs"),
        name="test_eval",
    )
    names = m.names if hasattr(m, "names") else dict(enumerate(CLASSES))
    per_class = {
        names[int(c)]: round(float(v), 4) for c, v in zip(m.box.ap_class_index, m.box.maps[m.box.ap_class_index])
    }
    write_json(
        RESULTS / "detector_eval.json",
        {
            "model": f"{args.weights} fine-tuned",
            "dataset": "Open Images V7 weapon subset via FiftyOne",
            "classes": CLASSES,
            "split_counts": counts,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "test": {
                "mAP50": round(float(m.box.map50), 4),
                "mAP50_95": round(float(m.box.map), 4),
                "precision": round(float(m.box.mp), 4),
                "recall": round(float(m.box.mr), 4),
                "per_class_mAP50_95": per_class,
            },
            "train_seconds": round(train_s, 1),
        },
        merge=False,
    )
    write_json(
        RESULTS / "data_summary.json",
        {
            "open_images_weapons": {
                "source": "Open Images V7 (images CC BY 2.0 per Flickr owner; annotations CC BY 4.0)",
                "classes": CLASSES,
                "counts": counts,
                "seed": SEED,
            }
        },
    )
    print("test mAP50", m.box.map50, "mAP50-95", m.box.map)


if __name__ == "__main__":
    main()
