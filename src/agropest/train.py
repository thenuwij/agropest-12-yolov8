"""Train YOLOv8s on AgroPest-12.

Usage
-----
    python -m agropest.train --data data/data.yaml
    python -m agropest.train --data data/data.yaml --epochs 30 --imgsz 640
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ultralytics import YOLO

from .config import DATA_DIR, RESULTS_DIR, TrainConfig


def train(data_yaml: Path | str, cfg: TrainConfig) -> tuple[YOLO, float, Path]:
    """Fine-tune a pretrained YOLOv8 checkpoint and return (model, minutes, run_dir)."""
    model = YOLO(cfg.weights)

    start = time.time()
    results = model.train(**cfg.to_ultralytics_kwargs(data_yaml))
    minutes = (time.time() - start) / 60

    run_dir = Path(results.save_dir)
    print(f"\nTraining finished in {minutes:.1f} min, artefacts in {run_dir}")
    print(f"Best checkpoint: {run_dir / 'weights' / 'best.pt'}")
    return model, minutes, run_dir


def parse_args() -> argparse.Namespace:
    cfg = TrainConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_DIR / "data.yaml")
    parser.add_argument("--weights", default=cfg.weights)
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--imgsz", type=int, default=cfg.imgsz)
    parser.add_argument("--batch", type=int, default=cfg.batch)
    parser.add_argument("--patience", type=int, default=cfg.patience)
    parser.add_argument("--device", default=cfg.device, help='CUDA index or "cpu"')
    parser.add_argument("--name", default=cfg.name)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        weights=args.weights,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        name=args.name,
    )

    if not Path(args.data).exists():
        raise SystemExit(
            f"{args.data} not found. Run `python -m agropest.data` first "
            "(see data/README.md)."
        )

    _, minutes, run_dir = train(args.data, cfg)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "train_summary.json").write_text(
        json.dumps(
            {"training_time_minutes": round(minutes, 2), "run_dir": str(run_dir)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
