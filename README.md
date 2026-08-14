# AgroPest-12: Crop Pest Detection with YOLOv8s

> Detecting and classifying 12 species of crop pests in field photography, and
> benchmarking a modern one-stage detector against a two-stage detector and
> classical computer-vision baselines.

<p>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c">
  <img alt="Ultralytics" src="https://img.shields.io/badge/Ultralytics-YOLOv8-00b8d4">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
</p>

Indiscriminate pesticide use harms human health, beneficial insects and the
crops themselves. Knowing _which_ pest is present, and _where_, lets growers
target treatment precisely. This project asks a concrete question: **how much
does detection accuracy actually improve when you move from handcrafted
features to a modern detector — and what does it cost in compute?**

Answer: **classification accuracy goes from 28% to 92%**, for roughly 4.5× the
training time, at a latency that still supports real-time monitoring (64 FPS).

---

## Headline results

Evaluated on a held-out test split of 546 images / 689 annotated instances.

| Model                                | Type             | Cls. accuracy | Cls. F1   | mAP@0.5   | Train time |
| ------------------------------------ | ---------------- | ------------- | --------- | --------- | ---------- |
| Random Forest (HOG+LBP+colour → PCA) | Classical        | 0.152         | ≈0.13     | ≈0.06     | ≈20 min    |
| Linear SVM (HOG+LBP+colour → PCA)    | Classical        | 0.282         | ≈0.28     | ≈0.17     | ≈30 min    |
| Faster R-CNN (ResNet50-FPN)          | Two-stage DL     | 0.764         | 0.609     | 0.714     | 113.5 min  |
| **YOLOv8s**                          | **One-stage DL** | **0.917**     | **0.918** | **0.797** | 135.6 min  |

Values marked ≈ are read from the comparison charts in the report; all others
are exact. Full numbers in [`results/reported_metrics.csv`](results/reported_metrics.csv).

<p align="center">
  <img src="results/figures/06_comparison_classification.png" width="46%">
  <img src="results/figures/07_comparison_detection.png" width="46%">
</p>

### YOLOv8s in detail

| Metric                              | Value                                                    |
| ----------------------------------- | -------------------------------------------------------- |
| mAP@0.5                             | **0.7966**                                               |
| mAP@0.5:0.95                        | 0.4856                                                   |
| Detection precision / recall        | 0.8618 / 0.7359                                          |
| Classification accuracy / F1        | 0.9173 / 0.9179                                          |
| Classification PR-AUC (one-vs-rest) | 0.9212                                                   |
| Inference latency                   | 15.69 ms/image (**63.7 FPS**)                            |
| Training                            | 50 epochs configured, early-stopped ≈ epoch 30, Colab T4 |

**Reading the gap between the two headline numbers.** Detection PR-AUC (0.797)
sits well below classification PR-AUC (0.921) because detection scores
localisation _and_ labelling together. Once the model has found an insect it
names it correctly ~92% of the time; drawing a tight box around a 20-pixel
thrip against cluttered foliage is the harder half of the problem. That is also
why mAP falls off sharply at stricter IoU thresholds (0.797 → 0.486).

<p align="center">
  <img src="results/figures/04_yolov8s_qualitative_predictions.png" width="70%">
</p>

---

## What's in this repository

```
.
├── src/agropest/            # The YOLOv8 pipeline, as importable modules
│   ├── config.py            #   All hyper-parameters in one dataclass
│   ├── data.py              #   Dataset download + data.yaml generation
│   ├── train.py             #   `python -m agropest.train`
│   ├── evaluate.py          #   Detection, classification and speed metrics
│   └── visualize.py         #   Confusion matrix, per-class AP, prediction grids
├── notebooks/
│   └── agropest_yolov8.ipynb    # End-to-end Colab walkthrough
├── data/
│   ├── data.yaml            # YOLO dataset descriptor (12 classes)
│   └── README.md            # How to obtain the dataset
├── results/
│   ├── reported_metrics.csv # Cross-model comparison
│   └── figures/             # All figures from the study
└── docs/
    └── report.pdf           # Full academic write-up
```

---

## Quickstart

```bash
git clone https://github.com/thenuwij/agropest-12-yolov8.git
cd agropest-12-yolov8

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src

# 1. Fetch the dataset and write data/data.yaml (see data/README.md)
python -m agropest.data --out data/data.yaml

# 2. Train
python -m agropest.train --data data/data.yaml --epochs 50 --imgsz 512

# 3. Evaluate — writes results/metrics.json and results/metrics.csv
python -m agropest.evaluate \
    --weights runs/agropest/yolov8s/weights/best.pt \
    --data data/data.yaml --split test

# 4. Figures
python -m agropest.visualize \
    --weights runs/agropest/yolov8s/weights/best.pt \
    --data data/data.yaml
```

No GPU? Pass `--device cpu`. Training on CPU is impractical, but evaluation of
a downloaded checkpoint is fine.

---

## Approach

### Why YOLOv8s

AgroPest-12 is a small-object problem: insects are often a few dozen pixels
across, partially occluded by leaves, and low-contrast against vegetation.
Three YOLOv8 design choices matter here.

- **Anchor-free head.** Earlier YOLO versions predicted offsets from predefined
  anchor boxes, which forces a hyper-parameter that fits some object scales and
  not others. Removing anchors lets the model localise across a wide scale range
  without that tuning burden.
- **C2f backbone blocks.** A lighter restructuring of the C3 block with better
  gradient flow and feature reuse, which helps preserve the fine edge detail
  that distinguishes an antenna from a leaf vein.
- **Decoupled head.** Classification and box regression are refined on separate
  branches instead of competing in one shared output.

The `s` variant was chosen over `m`/`l` deliberately: it fits comfortably in a
Colab T4's memory, and single-stage inference at 64 FPS is what makes the model
plausible for in-field monitoring rather than offline batch scoring.

### Training setup

|                |                                                            |
| -------------- | ---------------------------------------------------------- |
| Initialisation | `yolov8s.pt` (COCO-pretrained)                             |
| Input size     | 512 × 512                                                  |
| Batch size     | 16                                                         |
| Optimiser      | Adam (Ultralytics default)                                 |
| Losses         | CIoU (box), BCE (classification + objectness)              |
| Augmentation   | Mosaic, horizontal flip, HSV jitter (h .015 / s .7 / v .4) |
| Early stopping | patience 15 on validation mAP                              |
| Hardware       | NVIDIA Tesla T4 (Google Colab)                             |

512 × 512 was a hardware compromise, not an optimum — see [Limitations](#limitations).

### Evaluation

Detection is scored with the standard COCO-style evaluator. But the classical
baselines only ever emit a class label, so to compare like-for-like the detector
is also scored as a **classifier**: for each test image the highest-confidence
box supplies the predicted label, which is checked against the image's ground
truth. Images with no detection above threshold are excluded and counted
separately (14 of 546 here). This is implemented in
[`evaluate.py`](src/agropest/evaluate.py) and is the source of the accuracy
column in the headline table.

---

## Per-class results

Detection AP is strongest for species with distinctive silhouettes and weakest
for the small, low-contrast ones that blend into vegetation.

| Class        | Detection P | Detection R | Cls. F1 | Cls. support |
| ------------ | ----------- | ----------- | ------- | ------------ |
| Ants         | 0.924       | 0.695       | 0.972   | 54           |
| Bees         | 0.902       | 0.864       | 0.963   | 40           |
| Beetles      | 0.779       | 0.614       | 0.786   | 41           |
| Caterpillars | 0.847       | 0.495       | 0.825   | 41           |
| Earthworms   | 0.668       | 0.452       | 0.930   | 23           |
| Earwigs      | 0.863       | 0.688       | 0.926   | 57           |
| Grasshoppers | 0.826       | 0.618       | 0.900   | 38           |
| Moths        | 1.000       | 0.951       | 0.967   | 46           |
| Slugs        | 0.697       | 0.667       | 0.839   | 44           |
| Snails       | 0.913       | 0.838       | 0.943   | 44           |
| Wasps        | 0.958       | 0.973       | 0.968   | 46           |
| Weevils      | 0.966       | 0.977       | 0.957   | 58           |

The pattern is consistent across both columns: `Beetles`, `Caterpillars` and
`Earthworms` are the weak classes. Grad-CAM (below) explains why — the model
distinguishes beetles by leg groupings and short antennae and weevils by dorsal
texture, so front and rear views of the two collapse together. Earthworms and
slugs confuse for the same reason in reverse: both are segmented, limbless and
textureless.

### Model interpretability

Grad-CAM heatmaps over the backbone confirm the model attends to the same cues
a human entomologist would: ant leg joints and head/thorax junctions,
caterpillar dorsal hairs, grasshopper antennae and hind legs, snail shell
spirals.

<p align="center">
  <img src="results/figures/05_gradcam_yolov8s.png" width="70%">
</p>

---

## A note on class labels

While preparing this repository I found a bug in the original submission's YOLO
code: the class-name list was hard-coded from a _different_ pest dataset
(`aphids, armyworm, beetle, …`) rather than read from the dataset's own
`data.yaml` (`Ants, Bees, Beetles, …`).

Because YOLO labels are integer indices, this affected **display names only** —
every reported metric, box and index is unaffected, and the mapping is a clean
one-to-one at each index. It is nonetheless why the qualitative figure above
shows an ant captioned `aphids` and a snail captioned `termite`.

The per-class table above uses the **corrected** names. The code in
`src/agropest/` reads class names from `data.yaml` at runtime
(`agropest.data.load_class_names`) so the failure mode cannot recur.

---

## Limitations

Stated plainly, because they shaped the results:

- **Input resolution.** 512 × 512 was chosen to fit a free-tier Colab T4. YOLO
  downsamples early, so the smallest insects occupy very few feature-map pixels.
  This is the single most likely cause of the recall gap (0.736 vs. precision
  0.862) and of the mAP drop at stricter IoU.
- **No hyper-parameter search.** A single training run took over two hours on
  the available hardware, which made any meaningful sweep impractical.
- **Cross-model comparability.** The four models were trained by different team
  members on different hardware, so training-time comparisons are indicative
  rather than controlled. The accuracy comparison is sound — all models were
  evaluated on the same held-out test split.
- **Classical baselines were RAM-bound.** Full-dataset SVM training exceeded
  Colab's memory computing the kernel matrix, so the classical pipeline ran on a
  subsample. Its 28% accuracy is a floor, not a fair ceiling for the method.

## Future work

- Train at 640 × 640 or 1024 × 1024 and measure the recall recovery directly.
- Copy-paste and oversampling augmentation for the weak, small-object classes.
- Attention/feature-fusion blocks (SOD-YOLO-style) targeted at small objects.
- Noise-robustness study: rotation, hue/saturation shift, deliberately corrupted labels.

---

## Context and attribution

Built for **COMP9517 Computer Vision (25T3), UNSW Sydney** — group project.

The study compares four methods; this repository contains **my contribution, the
YOLOv8s detection pipeline** (design, training, evaluation, interpretability
analysis, and the results reported in the YOLO sections). The full report in
[`docs/report.pdf`](docs/report.pdf) is joint work, and the classical ML (SVM /
Random Forest) and Faster R-CNN pipelines were implemented by teammates — their
results are reproduced here for comparison and remain credited to them.

## References

Key works underpinning the approach; full bibliography in the report.

1. Ultralytics, _YOLOv8_, 2023. https://docs.ultralytics.com/models/yolov8/
2. Z. Tian et al., "FCOS: Fully Convolutional One-Stage Object Detection," _ICCV_, 2019.
3. S. Ren et al., "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks," _NeurIPS_, 2015.
4. T.-Y. Lin et al., "Feature Pyramid Networks for Object Detection," _CVPR_, 2017.
5. B. Khalili and A. W. Smyth, "SOD-YOLOv8 — Enhancing YOLOv8 for Small Object Detection," _Sensors_, 2024.
6. R. R. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization," _ICCV_, 2017.
7. N. Dalal and B. Triggs, "Histograms of Oriented Gradients for Human Detection," _CVPR_, 2005.
8. R. Majumdar, _AgroPest-12: A 12-Class Image Dataset of Crop Insects and Pests_, Kaggle, 2025.

## Licence

Code released under the [MIT Licence](LICENSE). The AgroPest-12 dataset is
subject to its own licence and is not redistributed here.
