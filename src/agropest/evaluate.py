"""Evaluate a trained YOLOv8 model on AgroPest-12.

Produces the three families of numbers reported in the write-up:

1. **Detection** — mAP@0.5, mAP@0.5:0.95, precision, recall (Ultralytics' own
   COCO-style evaluator, overall and per class).
2. **Classification** — image-level top-1 accuracy / precision / recall / F1 and
   a one-vs-rest PR-AUC, so the detector can be compared like-for-like against
   the SVM and Random Forest baselines, which only ever produced a class label.
3. **Efficiency** — mean latency and FPS over a fixed sample of test images.

Usage
-----
    python -m agropest.evaluate --weights runs/agropest/yolov8s/weights/best.pt \
                                --data data/data.yaml --split test
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import label_binarize
from ultralytics import YOLO

from .config import DATA_DIR, RESULTS_DIR, EvalConfig
from .data import load_class_names


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
def detection_metrics(model: YOLO, data_yaml: Path, split: str, names: list[str]) -> dict:
    """Run the Ultralytics validator and flatten its output into plain dicts."""
    m = model.val(data=str(data_yaml), split=split, verbose=False)

    overall = {
        "mAP@0.5": float(m.box.map50),
        "mAP@0.5:0.95": float(m.box.map),
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
    }

    per_class = {}
    for idx, class_id in enumerate(m.box.ap_class_index):
        per_class[names[int(class_id)]] = {
            "precision": float(m.box.p[idx]),
            "recall": float(m.box.r[idx]),
            "AP@0.5": float(m.box.ap50[idx]),
            "AP@0.5:0.95": float(m.box.ap[idx]),
        }

    return {"overall": overall, "per_class": per_class}


# --------------------------------------------------------------------------- #
# Classification (image-level top-1)
# --------------------------------------------------------------------------- #
def _image_true_label(label_path: Path) -> int | None:
    """Majority ground-truth class for one image, or None if it has no labels."""
    if not label_path.exists():
        return None
    classes = [
        int(line.split()[0]) for line in label_path.read_text().splitlines() if line.strip()
    ]
    if not classes:
        return None
    return max(set(classes), key=classes.count)


def classification_metrics(
    model: YOLO,
    split_dir: Path,
    names: list[str],
    conf: float = 0.25,
) -> dict:
    """Image-level top-1 classification, mirroring the baselines' framing.

    For each test image the highest-confidence detection supplies the predicted
    class; images with no detection above ``conf`` are excluded (and counted),
    exactly as in the reported experiments.
    """
    image_dir, label_dir = split_dir / "images", split_dir / "labels"

    y_true: list[int] = []
    y_pred: list[int] = []
    y_score: list[np.ndarray] = []
    n_missed = 0

    for image_path in sorted(image_dir.glob("*")):
        true = _image_true_label(label_dir / f"{image_path.stem}.txt")
        if true is None:
            continue

        boxes = model.predict(str(image_path), conf=conf, verbose=False)[0].boxes
        if len(boxes) == 0:
            n_missed += 1
            continue

        confidences = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        best = int(np.argmax(confidences))

        # One score per class = the strongest box of that class in the image.
        scores = np.zeros(len(names), dtype=float)
        for cls, cnf in zip(classes, confidences):
            scores[cls] = max(scores[cls], float(cnf))

        y_true.append(true)
        y_pred.append(int(classes[best]))
        y_score.append(scores)

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    y_score_arr = np.asarray(y_score)

    labels = list(range(len(names)))
    binarised = label_binarize(y_true_arr, classes=labels)
    pr_auc = float(average_precision_score(binarised, y_score_arr, average="macro"))

    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(
            precision_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)
        ),
        "recall": float(
            recall_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)
        ),
        "f1": float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)),
        "pr_auc_ovr": pr_auc,
        "n_evaluated": int(len(y_true_arr)),
        "n_no_detection": n_missed,
        "report": classification_report(
            y_true_arr,
            y_pred_arr,
            labels=labels,
            target_names=names,
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(y_true_arr, y_pred_arr, labels=labels).tolist(),
    }


# --------------------------------------------------------------------------- #
# Efficiency
# --------------------------------------------------------------------------- #
def inference_speed(model: YOLO, split_dir: Path, n_samples: int = 50) -> dict:
    """Mean end-to-end latency per image, after a short warm-up."""
    images = sorted((split_dir / "images").glob("*"))[:n_samples]
    if not images:
        raise FileNotFoundError(f"No images under {split_dir / 'images'}")

    for image_path in images[:3]:  # warm up CUDA kernels / autocast caches
        model.predict(str(image_path), verbose=False)

    timings = []
    for image_path in images:
        start = time.perf_counter()
        model.predict(str(image_path), verbose=False)
        timings.append(time.perf_counter() - start)

    mean = float(np.mean(timings))
    return {
        "n_images": len(images),
        "mean_latency_ms": mean * 1000,
        "fps": 1.0 / mean,
    }


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def evaluate(weights: Path, data_yaml: Path, cfg: EvalConfig) -> dict:
    names = load_class_names(data_yaml)
    dataset_cfg = yaml.safe_load(Path(data_yaml).read_text())
    split_key = {"test": "test", "val": "val", "valid": "val"}[cfg.split]
    split_dir = (Path(dataset_cfg["path"]) / dataset_cfg[split_key]).parent

    model = YOLO(str(weights))

    results = {
        "weights": str(weights),
        "split": cfg.split,
        "detection": detection_metrics(model, data_yaml, cfg.split, names),
        "classification": classification_metrics(model, split_dir, names, cfg.conf),
        "efficiency": inference_speed(model, split_dir, cfg.timing_samples),
    }

    det = results["detection"]["overall"]
    clf = results["classification"]
    eff = results["efficiency"]
    print("\nDetection      mAP@0.5 {mAP@0.5:.4f} | mAP@0.5:0.95 {mAP@0.5:0.95:.4f} | "
          "P {precision:.4f} | R {recall:.4f}".format(**det))
    print(f"Classification acc {clf['accuracy']:.4f} | F1 {clf['f1']:.4f} | "
          f"PR-AUC {clf['pr_auc_ovr']:.4f} | n={clf['n_evaluated']}")
    print(f"Efficiency     {eff['mean_latency_ms']:.2f} ms/image | {eff['fps']:.1f} FPS")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DATA_DIR / "data.yaml")
    parser.add_argument("--split", default="test", choices=["test", "val", "valid"])
    parser.add_argument("--conf", type=float, default=EvalConfig.conf)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "metrics.json")
    args = parser.parse_args()

    cfg = EvalConfig(split=args.split, conf=args.conf)
    results = evaluate(args.weights, args.data, cfg)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))

    # Flat one-row CSV for cross-model comparison with the other team methods.
    flat = {
        "method": "YOLOv8s",
        **{f"det_{k}": v for k, v in results["detection"]["overall"].items()},
        "cls_accuracy": results["classification"]["accuracy"],
        "cls_precision": results["classification"]["precision"],
        "cls_recall": results["classification"]["recall"],
        "cls_f1": results["classification"]["f1"],
        "cls_pr_auc": results["classification"]["pr_auc_ovr"],
        "latency_ms": results["efficiency"]["mean_latency_ms"],
        "fps": results["efficiency"]["fps"],
    }
    pd.DataFrame([flat]).to_csv(args.out.with_suffix(".csv"), index=False)
    print(f"\nWrote {args.out} and {args.out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
