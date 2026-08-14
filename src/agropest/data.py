"""Dataset acquisition and YOLO ``data.yaml`` handling for AgroPest-12.

The dataset already ships in YOLO format (``train|valid|test`` each containing
``images/`` and ``labels/``), so the only preparation step is pointing a
``data.yaml`` at wherever the archive was extracted.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .config import DATA_DIR, KAGGLE_DATASET


def download_dataset(dest: Path | None = None) -> Path:
    """Download AgroPest-12 via ``kagglehub`` and return the extracted root.

    Requires Kaggle credentials (``~/.kaggle/kaggle.json`` or the
    ``KAGGLE_USERNAME`` / ``KAGGLE_KEY`` environment variables).
    """
    import kagglehub

    path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    if dest is not None:
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.symlink_to(path, target_is_directory=True)
        return dest
    return path


def load_class_names(data_yaml: Path | str) -> list[str]:
    """Read the ordered class list out of a YOLO ``data.yaml``.

    Class names are always read from the dataset rather than hard-coded, so the
    label indices written in the ``.txt`` files and the names used in figures
    can never drift apart.
    """
    cfg = yaml.safe_load(Path(data_yaml).read_text())
    names = cfg["names"]
    if isinstance(names, dict):  # Ultralytics also accepts {index: name}
        return [names[i] for i in sorted(names)]
    return list(names)


def write_data_yaml(
    dataset_root: Path | str,
    class_names: list[str],
    out_path: Path | str = DATA_DIR / "data.yaml",
) -> Path:
    """Write a YOLO ``data.yaml`` pointing at ``dataset_root``."""
    dataset_root = Path(dataset_root).resolve()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    config = {
        "path": str(dataset_root),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": class_names,
    }
    out_path.write_text(yaml.dump(config, sort_keys=False))
    return out_path


def summarise_split(dataset_root: Path | str, split: str) -> dict:
    """Count images and per-class label instances in one split."""
    split_dir = Path(dataset_root) / split
    images = sorted((split_dir / "images").glob("*"))
    counts: dict[int, int] = {}
    for label_file in (split_dir / "labels").glob("*.txt"):
        for line in label_file.read_text().splitlines():
            if line.strip():
                cls = int(line.split()[0])
                counts[cls] = counts.get(cls, 0) + 1
    return {"split": split, "n_images": len(images), "instances_per_class": counts}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the AgroPest-12 dataset.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Path to an already-extracted dataset. Omit to download from Kaggle.",
    )
    parser.add_argument(
        "--out", type=Path, default=DATA_DIR / "data.yaml", help="Where to write data.yaml."
    )
    parser.add_argument(
        "--names-from",
        type=Path,
        default=DATA_DIR / "data.yaml",
        help="Existing data.yaml to copy the class list from.",
    )
    args = parser.parse_args()

    root = args.dataset_root or download_dataset()
    names = load_class_names(args.names_from)
    out = write_data_yaml(root, names, args.out)

    print(f"Dataset root : {root}")
    print(f"Wrote        : {out}")
    for split in ("train", "valid", "test"):
        if (Path(root) / split).exists():
            info = summarise_split(root, split)
            print(f"  {info['split']:<6} {info['n_images']:>6} images")


if __name__ == "__main__":
    main()
