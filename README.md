The demo code for our ESSD paper: Wang, Y., Chen, G., Yang, J., Gui, Z., and Peng, D.: A submesoscale eddy identification dataset in the northwest Pacific Ocean derived from GOCI I chlorophyll a data based on deep learning, Earth Syst. Sci. Data, 16, 5737–5752, https://doi.org/10.5194/essd-16-5737-2024, 2024.
# Submesoscale Eddy Identification from GOCI-I Chlorophyll-a (YOLOv7)

This repository contains the code accompanying the paper:

> Wang, Y., Chen, G., Yang, J., Gui, Z., and Peng, D. (2024): *A submesoscale eddy identification dataset in the northwest Pacific Ocean derived from GOCI I chlorophyll a data based on deep learning*. Earth System Science Data, 16, 5737–5752. https://doi.org/10.5194/essd-16-5737-2024

The project uses YOLOv7 to detect **submesoscale ocean eddies** from GOCI-I chlorophyll-a imagery. It supports:

- Training (`train.py`)
- Single-image / video inference (`predict.py`)
- Batch tile inference with daily JSON export (`dir_predict` mode)
- mAP evaluation (`get_map.py`)
- Anchor clustering (`kmeans_for_anchors.py`)

---

## 1. Project structure

```text
.
├── train.py                     # Training entry point
├── predict.py                   # Inference entry point (multi-mode)
├── get_map.py                   # VOC-style mAP evaluation
├── kmeans_for_anchors.py        # Anchor clustering
├── yolo.py                      # YOLO inference wrapper
├── nets/                        # Network architecture and losses
├── utils/                       # Data loading, training, and postprocessing utilities
├── Preprocessing_training_set/  # Dataset preprocessing scripts
├── model_data/                  # Classes, anchors, weights, fonts, etc.
└── img/                         # Example input files
```

---

## 2. Requirements

Recommended Python version: **3.8+**.

Install dependencies:

```bash
pip install -r requirements.txt
```

Notes:

- Training/inference is configured for PyTorch + CUDA by default.
- You can disable CUDA in code/config if no GPU is available.
- `dir_predict` mode in `predict.py` uses `torchvision.ops.nms`.

---

## 3. Dataset preparation (training)

The training workflow uses a VOC-style dataset organization (see `VOCdevkit` assumptions in scripts):

1. Prepare images and XML annotations.
2. Generate train/val split files (e.g., `2007_train.txt`, `2007_val.txt`).
3. Define class names in `model_data/eddy.txt` (e.g., `AE`, `CE`).

If needed, use scripts in `Preprocessing_training_set/` for tiling, augmentation, and VOC file list generation.

---

## 4. How to run

### 4.1 Training

```bash
python train.py
```

Before training, check key parameters in `train.py`, including:

- `model_path` (pretrained weights)
- `classes_path` (class file)
- `input_shape` (input resolution)
- `batch_size`, `Freeze_Train`, `UnFreeze_Epoch`
- dataset root path (VOC directory)

---

### 4.2 Inference

```bash
python predict.py
```

Switch behavior via the `mode` variable in `predict.py`:

- `predict`: single image
- `video`: webcam/video stream
- `fps`: speed benchmark
- `dir_predict`: folder batch inference with daily tile fusion and JSON export
- `heatmap`: heatmap visualization
- `export_onnx`: export ONNX model

> For `dir_predict`, update local input/output directories to your own data paths.

---

### 4.3 mAP evaluation

```bash
python get_map.py
```

Configure script settings for:

- ground-truth / prediction paths
- IoU and related thresholds
- plotting/visualization options

---

## 5. Engineering updates in this codebase

To improve maintainability and reproducibility:

1. **`predict.py` refactor**
   - Split geogrid loading, coordinate projection, date iteration, and directory inference into independent functions.
   - Fixed type mismatch risk in tile row/column index comparisons (`int` vs `str`).
   - Unified path handling via `os.path`.
   - Added clearer function-level comments/docstrings.

2. **README rewrite**
   - Added paper context, project structure, requirements, and run instructions.
   - Clarified expected data organization for `dir_predict`.

---

## 6. Daily JSON output contents (`dir_predict`)

Per-day JSON files include:

- date
- AE/CE counts
- pixel-space bounding boxes
- eddy center longitude/latitude
- bounding-box corner longitude/latitude
- inscribed radius and ellipse area estimates
- confidence values

This aligns with the production workflow used to build the eddy identification dataset reported in the paper.

---

## 7. Citation

If you use this repository, please cite:

```bibtex
@article{wang2024submesoscale,
  author  = {Wang, Y. and Chen, G. and Yang, J. and Gui, Z. and Peng, D.},
  title   = {A submesoscale eddy identification dataset in the northwest Pacific Ocean derived from GOCI I chlorophyll a data based on deep learning},
  journal = {Earth System Science Data},
  volume  = {16},
  pages   = {5737--5752},
  year    = {2024},
  doi     = {10.5194/essd-16-5737-2024}
}
```

---

## 8. License

See `LICENSE` for details.
