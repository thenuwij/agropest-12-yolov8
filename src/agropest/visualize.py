"""Figures: confusion matrix, per-class AP bars and qualitative prediction grids.

Usage
-----
    python -m agropest.visualize --weights runs/agropest/yolov8s/weights/best.pt \
                                 --metrics results/metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe; must precede pyplot import

import matplotlib.pyplot as plt
import numpy as np
import yaml
from ultralytics import YOLO

from .config import DATA_DIR, FIGURES_DIR, RESULTS_DIR
from .data import load_class_names


def plot_confusion_matrix(matrix: np.ndarray, names: list[str], out: Path) -> Path:
    """Row-normalised confusion matrix over image-level top-1 predictions."""
    matrix = np.asarray(matrix, dtype=float)
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalised = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(normalised, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("YOLOv8s image-level confusion matrix (test set)")

    for i in range(len(names)):
        for j in range(len(names)):
            if normalised[i, j] > 0.01:
                ax.text(
                    j, i, f"{normalised[i, j]:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="white" if normalised[i, j] > 0.5 else "black",
                )

    fig.colorbar(im, ax=ax, fraction=0.046, label="proportion of true class")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_per_class_ap(per_class: dict, out: Path) -> Path:
    """Horizontal bar chart of AP@0.5 per class, sorted worst to best."""
    items = sorted(per_class.items(), key=lambda kv: kv[1]["AP@0.5"])
    labels = [name for name, _ in items]
    values = [stats["AP@0.5"] for _, stats in items]

    fig, ax = plt.subplots(figsize=(8, 0.45 * len(labels) + 1.5))
    ax.barh(labels, values, color="#2f6f4e")
    ax.set_xlim(0, 1)
    ax.set_xlabel("AP@0.5")
    ax.set_title("YOLOv8s per-class detection AP (test set)")
    ax.bar_label(ax.containers[0], fmt="%.3f", padding=3, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_predictions(model: YOLO, split_dir: Path, out: Path, n: int = 9) -> Path:
    """Grid of annotated predictions on held-out images."""
    images = sorted((split_dir / "images").glob("*"))[:n]
    side = int(np.ceil(np.sqrt(len(images))))

    fig, axes = plt.subplots(side, side, figsize=(4 * side, 4 * side))
    for ax, image_path in zip(np.ravel(axes), images):
        annotated = model.predict(str(image_path), verbose=False)[0].plot()
        ax.imshow(annotated[:, :, ::-1])  # Ultralytics returns BGR
        ax.set_title(image_path.name, fontsize=8)
    for ax in np.ravel(axes):
        ax.axis("off")

    fig.suptitle("YOLOv8s predictions on unseen test images", fontsize=14)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DATA_DIR / "data.yaml")
    parser.add_argument("--metrics", type=Path, default=RESULTS_DIR / "metrics.json")
    parser.add_argument("--outdir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args()

    names = load_class_names(args.data)
    dataset_cfg = yaml.safe_load(args.data.read_text())
    split_dir = (Path(dataset_cfg["path"]) / dataset_cfg["test"]).parent

    written = []
    if args.metrics.exists():
        metrics = json.loads(args.metrics.read_text())
        written.append(
            plot_confusion_matrix(
                metrics["classification"]["confusion_matrix"],
                names,
                args.outdir / "confusion_matrix.png",
            )
        )
        written.append(
            plot_per_class_ap(
                metrics["detection"]["per_class"], args.outdir / "per_class_ap.png"
            )
        )
    else:
        print(f"{args.metrics} not found, skipping metric plots.")

    written.append(
        plot_predictions(
            YOLO(str(args.weights)), split_dir, args.outdir / "predictions.png"
        )
    )

    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
