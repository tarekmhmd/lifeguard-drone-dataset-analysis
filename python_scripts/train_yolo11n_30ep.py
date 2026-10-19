"""
Train YOLO11n on SeaDronesSee_ODv2 (30 epochs), then validate and export metrics.

Usage:
  python train_yolo11n_30ep.py               # full pipeline: preflight -> train -> validate -> export
  python train_yolo11n_30ep.py --skip-train  # validate an existing checkpoint only (no training)

Set PROJECT below to your project root (default E:/lifeguardDron).
Requires a CUDA GPU (Ultralytics + torch with CUDA), the dataset, and yolo11n.pt.
"""

import argparse
import csv
import math
import shutil
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Configuration -- EDIT THESE PATHS
# ---------------------------------------------------------------------------
PROJECT = Path("E:/lifeguardDron")
DATASET_YAML = PROJECT / "data_prepared" / "detection" / "SeaDronesSee_ODv2" / "dataset.yaml"
PRETRAINED_PT = PROJECT / "yolo11n.pt"   # original pretrained weights (NOT the smoke-test best.pt)
BASELINE_CSV = PROJECT / "reports" / "model_evaluation" / "yolo11n_validation_metrics.csv"
RUN_NAME = "seadrones_yolo11n_30ep_v1"
RUN_DIR = PROJECT / "runs" / "detect" / RUN_NAME
NEW_METRICS_CSV = PROJECT / "reports" / "model_evaluation" / "seadrones_yolo11n_30ep_v1_metrics.csv"

EXPECTED_NAMES = {0: "swimmer", 1: "boat", 2: "jetski", 3: "life_saving_appliances", 4: "buoy"}

TRAIN_CONFIG = {
    "epochs": 30,
    "patience": 10,
    "imgsz": 640,
    "batch": 4,
    "device": 0,
    "seed": 42,
    "deterministic": True,
    "pretrained": True,
    "optimizer": "auto",
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
    "mosaic": 1.0,
    "erasing": 0.4,
    "cls_remap": True,
    "project": str(PROJECT / "runs" / "detect"),
    "name": RUN_NAME,
    "exist_ok": False,
    "resume": False,
    "amp": True,
    "workers": 2,
    "val": True,
    "split": "val",
}


def require(cond, msg):
    if not cond:
        raise SystemExit("PRECHECK FAILED: " + msg)
    print("  OK  " + msg)


def load_names():
    cfg = yaml.safe_load(DATASET_YAML.read_text(encoding="utf-8"))
    names = cfg.get("names")
    if names != EXPECTED_NAMES:
        raise SystemExit("class mapping mismatch: %s" % names)
    return names


def preflight():
    print("=== Preflight ===")
    require(PROJECT.exists(), "project root exists")
    require(DATASET_YAML.exists(), "dataset.yaml exists")
    names = load_names()
    require(PRETRAINED_PT.exists(), "pretrained yolo11n.pt exists: " + str(PRETRAINED_PT))
    require(torch.cuda.is_available(), "CUDA is available")
    print("  GPU 0:", torch.cuda.get_device_name(0))
    free = shutil.disk_usage(PROJECT).free
    print("  free disk (GB): %.1f" % (free / 1e9))
    require(free > 2e9, ">= 2 GB free disk space")
    return names


def train():
    print("=== Training ===")
    model = YOLO(str(PRETRAINED_PT))
    model.train(**dict(TRAIN_CONFIG))
    best = RUN_DIR / "weights" / "best.pt"
    if not best.exists():
        raise SystemExit("training did not produce best.pt: " + str(best))
    print("Best checkpoint:", best)
    return best


def extract_per_class(metrics):
    """Map per-class metrics correctly using the 1-D ap/ap50 arrays.

    NOTE (bug fix): Ultralytics 8.4.182 exposes `metrics.box.map50` and
    `metrics.box.map` as SCALARS (mean over classes). The per-class arrays are
    `metrics.box.ap50` (mAP@0.5) and `metrics.box.ap` (mAP@0.5:0.95), aligned with
    `metrics.box.ap_class_index`. Indexing the scalar raised IndexError.
    """
    ap50 = metrics.box.ap50
    ap = metrics.box.ap
    ap_cls = list(map(int, metrics.box.ap_class_index))
    per = {}
    for idx, cls_id in enumerate(ap_cls):
        per[cls_id] = {
            "p": float(metrics.box.p[idx]),
            "r": float(metrics.box.r[idx]),
            "map50": float(ap50[idx]),
            "map5095": float(ap[idx]),
        }
    return per


def validate_and_export(best_pt, names):
    print("=== Validation ===")
    model = YOLO(str(best_pt))
    metrics = model.val(data=str(DATASET_YAML), imgsz=640, device=0, iou=0.7,
                        conf=0.001, split="val", plots=False, verbose=False)
    per = extract_per_class(metrics)
    names_list = [names[i] for i in range(len(names))]

    print("%-26s %10s %10s %10s %10s" % ("class", "P", "R", "mAP50", "mAP50-95"))
    new_metrics = {}
    for i in range(len(names)):
        m = per.get(i)
        if m is None or not all(math.isfinite(v) for v in m.values()):
            print("%-26s %10s %10s %10s %10s" % (names_list[i], "n/a", "n/a", "n/a", "n/a"))
            new_metrics[i] = None
        else:
            print("%-26s %10.4f %10.4f %10.4f %10.4f" % (
                names_list[i], m["p"], m["r"], m["map50"], m["map5095"]))
            new_metrics[i] = m

    print("=== Export ===")
    if NEW_METRICS_CSV.exists():
        with open(NEW_METRICS_CSV, encoding="utf-8") as f:
            existing = list(csv.reader(f))
        if [r for r in existing if r and r[0] != "Class"]:
            raise SystemExit("REFUSING to overwrite existing CSV with data: " + str(NEW_METRICS_CSV))
    NEW_METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(NEW_METRICS_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Class", "Precision", "Recall", "mAP50", "mAP50-95"])
        for i in range(len(names)):
            m = new_metrics.get(i)
            w.writerow([names_list[i],
                        "" if m is None else "%.6f" % m["p"],
                        "" if m is None else "%.6f" % m["r"],
                        "" if m is None else "%.6f" % m["map50"],
                        "" if m is None else "%.6f" % m["map5095"]])
    print("Wrote:", NEW_METRICS_CSV)
    return new_metrics


def main():
    parser = argparse.ArgumentParser(description="Train YOLO11n (30 epochs) and validate/export metrics.")
    parser.add_argument("--skip-train", action="store_true",
                        help="validate an existing checkpoint only (no training)")
    args = parser.parse_args()

    names = preflight()

    if args.skip_train:
        best = RUN_DIR / "weights" / "best.pt"
        if not best.exists():
            raise SystemExit("best.pt not found (nothing to validate): " + str(best))
    else:
        if RUN_DIR.exists():
            raise SystemExit("REFUSING to train: run dir already exists: " + str(RUN_DIR))
        best = train()

    validate_and_export(best, names)
    print("\nDone. Best checkpoint:", best)


if __name__ == "__main__":
    main()
