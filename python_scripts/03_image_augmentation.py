"""
SeaDronesSee_ODv2 data-augmentation pipeline (class-aware, conservative).

Usage:
  python 03_image_augmentation.py                 # full analysis + dry-run plan (no files written)
  python 03_image_augmentation.py --generate      # ACTUALLY write augmented images/labels (after review)

The augmentation targets the under-represented `life_saving_appliances` class (class 3)
using OpenCV/NumPy only (albumentations is not installed). It never modifies the
original dataset and only ever selects the training split.

Set PROJECT below to your project root (default E:/lifeguardDron).
"""

import argparse
import math
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Configuration -- EDIT THESE PATHS
# ---------------------------------------------------------------------------
PROJECT = Path("E:/lifeguardDron")
DATASET_YAML = PROJECT / "data_prepared" / "detection" / "SeaDronesSee_ODv2" / "dataset.yaml"
TRAIN_IMAGES = PROJECT / "data_prepared" / "detection" / "SeaDronesSee_ODv2" / "images" / "train"
TRAIN_LABELS = PROJECT / "data_prepared" / "detection" / "SeaDronesSee_ODv2" / "labels" / "train"
AUG_OUT_DIR = PROJECT / "data_prepared" / "augmentation" / "SeaDronesSee_ODv2_augmented_train"

NAMES = {0: "swimmer", 1: "boat", 2: "jetski", 3: "life_saving_appliances", 4: "buoy"}
TARGET = 3                       # life_saving_appliances
AUG_PER_IMAGE = 2                # conservative transforms per selected image
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

AUG_TRANSFORMS = [
    ("flip",   lambda im, b: flip_h(im, b)),
    ("rot10",  lambda im, b: rotate(im, b, 10)),
    ("bright", lambda im, b: (brightness_contrast(im, 1.15, 5), b)),
]

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def list_images(d):
    return sorted(p for p in d.glob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def list_labels(d):
    return sorted(p for p in d.glob("*.txt"))


def read_boxes(label_path):
    out = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) != 5:
            continue
        cls = int(float(p[0]))
        cx, cy, w, h = map(float, p[1:5])
        out.append((cls, cx, cy, w, h))
    return out


# ---------------------------------------------------------------------------
# Geometric helpers (bounding boxes stay synchronized)
# ---------------------------------------------------------------------------
def to_pixels(boxes, W, H):
    return [(c, (cx - w/2) * W, (cy - h/2) * H, (cx + w/2) * W, (cy + h/2) * H)
            for (c, cx, cy, w, h) in boxes]


def to_norm(boxes, W, H):
    return [(c, ((x1 + x2) / 2) / W, ((y1 + y2) / 2) / H, (x2 - x1) / W, (y2 - y1) / H)
            for (c, x1, y1, x2, y2) in boxes]


def clip_box(c, x1, y1, x2, y2, W, H, min_px=2):
    x1 = max(0.0, min(float(x1), W)); x2 = max(0.0, min(float(x2), W))
    y1 = max(0.0, min(float(y1), H)); y2 = max(0.0, min(float(y2), H))
    if (x2 - x1) < min_px or (y2 - y1) < min_px:
        return None
    return (c, x1, y1, x2, y2)


def flip_h(img, boxes):
    H, W = img.shape[:2]
    img2 = cv2.flip(img, 1)
    boxes2 = [b for b in (clip_box(c, W - x2, y1, W - x1, y2, W, H)
              for (c, x1, y1, x2, y2) in boxes) if b]
    return img2, boxes2


def rotate(img, boxes, angle):
    H, W = img.shape[:2]
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
    img2 = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    boxes2 = []
    for (c, x1, y1, x2, y2) in boxes:
        t = cv2.transform(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
                          .reshape(-1, 1, 2), M).reshape(-1, 2)
        b = clip_box(c, t[:, 0].min(), t[:, 1].min(), t[:, 0].max(), t[:, 1].max(), W, H)
        if b:
            boxes2.append(b)
    return img2, boxes2


def translate(img, boxes, dx, dy):
    H, W = img.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    img2 = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    boxes2 = [b for b in (clip_box(c, x1 + dx, y1 + dy, x2 + dx, y2 + dy, W, H)
              for (c, x1, y1, x2, y2) in boxes) if b]
    return img2, boxes2


def brightness_contrast(img, alpha=1.0, beta=0.0):
    # photometric only: box coordinates unchanged
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------
def verify_dataset():
    if not DATASET_YAML.exists():
        raise SystemExit("dataset.yaml not found: " + str(DATASET_YAML))
    cfg = yaml.safe_load(DATASET_YAML.read_text(encoding="utf-8"))
    if cfg.get("names") != NAMES:
        raise SystemExit("class mapping mismatch: %s" % cfg.get("names"))
    for p in (TRAIN_IMAGES, TRAIN_LABELS):
        if not (p.exists() and p.is_dir()):
            raise SystemExit("missing directory: " + str(p))
    print("  OK  dataset + class mapping verified")


def counts_and_validation():
    obj = Counter(); img_with = Counter(); n_img = 0
    empty_files, malformed, dup_rows = [], [], 0
    for lbl in list_labels(TRAIN_LABELS):
        n_img += 1
        lines = [l.strip() for l in lbl.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not lines:
            empty_files.append(lbl.name); continue
        seen = set(); classes = set()
        for ln in lines:
            parts = ln.split()
            if len(parts) != 5:
                malformed.append((lbl.name, ln)); continue
            cls = int(float(parts[0])); cx, cy, w, h = map(float, parts[1:5])
            if (cls not in NAMES or not all(math.isfinite(v) for v in (cx, cy, w, h))
                    or not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1)):
                malformed.append((lbl.name, ln)); continue
            obj[cls] += 1; classes.add(cls)
            key = tuple(round(float(x), 8) for x in parts)
            if key in seen: dup_rows += 1
            seen.add(key)
        for c in classes:
            img_with[c] += 1
    print("train images:", n_img)
    print("objects per class:", dict(obj))
    print("empty label files:", len(empty_files), "| malformed rows:", len(malformed),
          "| duplicate rows:", dup_rows)
    return obj


def class_aware_plan():
    c3_stems = sorted(lbl.stem for lbl in list_labels(TRAIN_LABELS)
                      if any(b[0] == TARGET for b in read_boxes(lbl)))
    img_map = {p.stem: p for p in list_images(TRAIN_IMAGES)}
    c3_before = sum(1 for lbl in list_labels(TRAIN_LABELS) for b in read_boxes(lbl) if b[0] == TARGET)
    gen_plan = []
    c3_after_est = c3_before
    for stem in c3_stems:
        ip = img_map.get(stem)
        img = cv2.imread(str(ip)) if ip else None
        if img is None:
            continue
        W, H = img.shape[1], img.shape[0]
        boxes = to_pixels(read_boxes(TRAIN_LABELS / (stem + ".txt")), W, H)
        for (tname, fn) in AUG_TRANSFORMS:
            _, boxes2 = fn(img, boxes)
            c3_after_est += sum(1 for b in boxes2 if b[0] == TARGET)
            gen_plan.append((stem, tname, fn))
    print("training images containing class 3:", len(c3_stems))
    print("class-3 instances before:", c3_before)
    print("class-3 instances after (estimated):", c3_after_est)
    print("planned augmented images:", len(gen_plan))
    return gen_plan, c3_before


def generate(gen_plan):
    if AUG_OUT_DIR.exists():
        raise SystemExit("REFUSING to overwrite existing output dir: " + str(AUG_OUT_DIR))
    AUG_OUT_DIR.mkdir(parents=True)
    (AUG_OUT_DIR / "images").mkdir(); (AUG_OUT_DIR / "labels").mkdir()
    img_map = {p.stem: p for p in list_images(TRAIN_IMAGES)}
    written = 0
    for (stem, tname, fn) in gen_plan:
        img = cv2.imread(str(img_map[stem]))
        W, H = img.shape[1], img.shape[0]
        boxes = to_pixels(read_boxes(TRAIN_LABELS / (stem + ".txt")), W, H)
        aug_img, aug_boxes = fn(img, boxes)
        out_stem = "%s_aug_%s_%03d" % (stem, tname, written)
        cv2.imwrite(str(AUG_OUT_DIR / "images" / (out_stem + ".jpg")), aug_img)
        with open(AUG_OUT_DIR / "labels" / (out_stem + ".txt"), "w", encoding="utf-8") as f:
            for (c, cx, cy, w, h) in to_norm(aug_boxes, W, H):
                f.write("%d %.6f %.6f %.6f %.6f\n" % (c, cx, cy, w, h))
        written += 1
    aug_imgs = set(p.stem for p in list_images(AUG_OUT_DIR / "images"))
    aug_lbls = set(p.stem for p in list_labels(AUG_OUT_DIR / "labels"))
    print("Generated images:", len(aug_imgs), "| labels:", len(aug_lbls),
          "| pairing ok:", aug_imgs == aug_lbls)


def main():
    parser = argparse.ArgumentParser(description="Class-aware augmentation for class 3.")
    parser.add_argument("--generate", action="store_true",
                        help="actually write augmented files (default is dry-run)")
    args = parser.parse_args()

    print("=== environment ===")
    print("OpenCV:", cv2.__version__, "| NumPy:", np.__version__)

    verify_dataset()
    counts_and_validation()
    gen_plan, c3_before = class_aware_plan()

    if not args.generate:
        print("\nDRY RUN: no files were written. Re-run with --generate to write files.")
        return

    generate(gen_plan)


if __name__ == "__main__":
    main()
