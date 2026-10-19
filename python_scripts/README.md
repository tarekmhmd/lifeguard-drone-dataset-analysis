# Python Scripts — SeaDronesSee_ODv2 Project

Standalone scripts extracted and refactored from the Jupyter notebooks in `../notebooks`.
They are intended to be run from a terminal (not a notebook).

## Setup

- **Python:** 3.11 (developed on 3.11.4).
- **Dependencies:** see `../requirements.txt` (`pip install -r requirements.txt`).
- **CUDA GPU:** required for `train_yolo11n_30ep.py` and `dataset_inspection.py --step evaluate|diagnostic`. The other steps are CPU/read-only.
- **External assets** (all under the project root, default `E:/lifeguardDron`):
  - Dataset: `data_prepared/detection/SeaDronesSee_ODv2/` (images + labels + `dataset.yaml`).
  - Pretrained weights: `yolo11n.pt`.
  - Trained checkpoint: `runs/detect/seadrones_smoke_test/weights/best.pt`.

## Configuration

Each script has a `Configuration` block near the top with a `PROJECT` path. Edit it if your
project root differs from `E:/lifeguardDron`.

## Scripts

| Script | Purpose | GPU? | Notes |
|---|---|---|---|
| `01_dataset_audit.py` | Verify project structure, dataset YAML, class mapping, environment, and per-class object counts. | No | Read-only. The source notebook also had exploratory train/val/predict cells, which were removed (covered elsewhere). |
| `dataset_inspection.py` | Dataset overview (counts, box size, validation), model evaluation (per-class metrics), and a raw-vs-post failure-mode diagnostic for class 3. | evaluate/diagnostic steps only | Run `--step overview` for a fast read-only check. |
| `03_image_augmentation.py` | Class-aware, conservative augmentation for `life_saving_appliances`. | No | Dry-run by default; `--generate` writes files. Never modifies the source dataset. |
| `train_yolo11n_30ep.py` | Train YOLO11n (30 epochs) then validate + export per-class metrics. | Yes | `--skip-train` validates an existing checkpoint only. |

## Run examples

```bash
cd python_scripts

# Fast, read-only dataset checks
python 01_dataset_audit.py
python dataset_inspection.py --step overview

# Model evaluation (requires the checkpoint)
python dataset_inspection.py --step evaluate

# Augmentation plan (dry-run) and generation
python 03_image_augmentation.py
python 03_image_augmentation.py --generate

# Training (hours) and validation-only
python train_yolo11n_30ep.py
python train_yolo11n_30ep.py --skip-train
```

## Known limitations

- The scripts preserve the original notebook logic (including one earlier metric-extraction
  bug that has now been corrected in `train_yolo11n_30ep.py` — see the `extract_per_class`
  docstring).
- `03_image_augmentation.py` does not include the visualization/preview cells from the
  notebook; it keeps the transform logic and the dry-run/generation pipeline only.
- `dataset_inspection.py` does not include the interactive label-audit visualization
  (matplotlib); those remain in the source notebook.
