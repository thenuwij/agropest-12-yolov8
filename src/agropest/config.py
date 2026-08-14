"""Central configuration for the AgroPest-12 YOLOv8 pipeline.

Every hyper-parameter used in the reported experiments lives here so that a
run can be reproduced from a single file rather than from scattered notebook
cells.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

# Repository root (…/src/agropest/config.py -> …/)
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

#: Kaggle slug for the dataset. See data/README.md for download instructions.
KAGGLE_DATASET = "rupankarmajumdar/crop-pests-dataset"


@dataclass
class TrainConfig:
    """Hyper-parameters for the YOLOv8s training run reported in the paper."""

    weights: str = "yolov8s.pt"
    epochs: int = 50
    imgsz: int = 512          # 512 chosen to fit a Colab T4; see README "Limitations"
    batch: int = 16
    patience: int = 15        # early stopping; triggered near epoch 30 in our run
    device: str = "0"         # "0" for the first CUDA device, "cpu" to force CPU
    seed: int = 0

    project: str = "runs/agropest"
    name: str = "yolov8s"
    exist_ok: bool = True

    # Built-in Ultralytics augmentations.
    hsv_h: float = 0.015      # hue jitter
    hsv_s: float = 0.7        # saturation jitter
    hsv_v: float = 0.4        # value/brightness jitter
    fliplr: float = 0.5       # horizontal flip probability
    mosaic: float = 1.0       # mosaic augmentation

    def to_ultralytics_kwargs(self, data_yaml: str | Path) -> dict:
        """Render this config as keyword arguments for ``YOLO.train``."""
        kwargs = asdict(self)
        kwargs.pop("weights")
        kwargs["data"] = str(data_yaml)
        kwargs["save"] = True
        kwargs["verbose"] = True
        return kwargs


@dataclass
class EvalConfig:
    """Settings for evaluation, metric extraction and timing."""

    split: str = "test"
    conf: float = 0.25              # confidence threshold for the classification view
    iou: float = 0.5                # NMS IoU threshold
    timing_samples: int = 50        # images used for the inference-speed benchmark
    qualitative_samples: int = 9    # images in the prediction grid figure
    figures_dir: Path = field(default_factory=lambda: FIGURES_DIR)
