"""
Dataset audit for the SeaDronesSee_ODv2 project.

Usage:
  python 01_dataset_audit.py

Checks the project structure, expected files, dataset YAML + class mapping, the
environment (PyTorch/CUDA), and per-class object counts. Read-only.

NOTE: the source notebook also contained exploratory training/inference cells
(`model.train`, `model.val`, `model.predict`). Those are NOT part of a dataset
audit and were removed here; training is covered by train_yolo11n_30ep.py and the
diagnostics are covered by dataset_inspection.py.

Set PROJECT below to your project root (default E:/lifeguardDron).
"""

import csv
import os
from collections import Counter
from pathlib import Path

import torch
import yaml

# ---------------------------------------------------------------------------
# Configuration -- EDIT THESE PATHS
# ---------------------------------------------------------------------------
PROJECT = Path("E:/lifeguardDron")
DATASET_YAML = PROJECT / "data_prepared" / "detection" / "SeaDronesSee_ODv2" / "dataset.yaml"
INVENTORY_CSV = PROJECT / "datasets_inventory.csv"
EXPECTED_NAMES = {0: "swimmer", 1: "boat", 2: "jetski", 3: "life_saving_appliances", 4: "buoy"}

EXPECTED_FILES = [
    "scripts/run_pipeline.py",
    "datasets_inventory.csv",
]


def check_environment():
    print("=== Environment ===")
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))


def check_project_structure():
    print("\n=== Project structure ===")
    print("Exists:", PROJECT.exists(), "| Path:", PROJECT)
    for rel in EXPECTED_FILES:
        p = PROJECT / rel
        print("[FOUND]   " + rel if p.is_file() else "[MISSING] " + rel)


def check_inventory():
    print("\n=== datasets_inventory.csv ===")
    if not INVENTORY_CSV.exists():
        print("File not found:", INVENTORY_CSV)
        return
    with INVENTORY_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        print("Columns:", reader.fieldnames)
        for i, row in enumerate(reader):
            if i >= 10:
                break
            print("Row %d: %s" % (i + 1, dict(row)))


def check_dataset():
    print("\n=== dataset.yaml ===")
    if not DATASET_YAML.exists():
        print("MISSING:", DATASET_YAML)
        return
    cfg = yaml.safe_load(DATASET_YAML.read_text(encoding="utf-8"))
    print("nc:", cfg.get("nc"), "| names:", cfg.get("names"))
    if cfg.get("names") != EXPECTED_NAMES:
        raise SystemExit("class mapping mismatch: %s" % cfg.get("names"))
    return cfg


def count_objects(cfg):
    print("\n=== Object counts per class ===")
    base = Path(cfg["path"])
    class_names = EXPECTED_NAMES
    for split in ("train", "val"):
        label_dir = base / "labels" / split
        counts = Counter()
        for f in label_dir.glob("*.txt"):
            for line in f.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if parts:
                    try:
                        counts[int(parts[0])] += 1
                    except ValueError:
                        pass
        print(split.upper())
        for cid, name in class_names.items():
            print("  %-26s %d" % (name, counts[cid]))


def main():
    check_environment()
    check_project_structure()
    check_inventory()
    cfg = check_dataset()
    count_objects(cfg)


if __name__ == "__main__":
    main()
