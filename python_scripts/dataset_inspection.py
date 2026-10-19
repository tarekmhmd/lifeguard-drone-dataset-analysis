"""
SeaDronesSee_ODv2 dataset inspection + model-evaluation diagnostics.

Usage:
  python dataset_inspection.py --step overview     # counts, size stats, label validation (fast, read-only)
  python dataset_inspection.py --step evaluate     # load best.pt, val, per-class metrics + export CSV
  python dataset_inspection.py --step diagnostic   # raw-vs-post failure-mode analysis for class 3 (GPU inference)
  python dataset_inspection.py --step all          # overview + evaluate + diagnostic

Set PROJECT below to your project root (default E:/lifeguardDron).
`evaluate` and `diagnostic` require a trained checkpoint and (for `diagnostic`) a GPU.
"""

import argparse
import csv
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Configuration -- EDIT THESE PATHS
# ---------------------------------------------------------------------------
PROJECT = Path("E:/lifeguardDron")
DATASET_DIR = PROJECT / "data_prepared" / "detection" / "SeaDronesSee_ODv2"
DATASET_YAML = DATASET_DIR / "dataset.yaml"
CHECKPOINT = PROJECT / "runs" / "detect" / "seadrones_smoke_test" / "weights" / "best.pt"
VAL_IMAGES = DATASET_DIR / "images" / "val"
VAL_LABELS = DATASET_DIR / "labels" / "val"
TRAIN_IMAGES = DATASET_DIR / "images" / "train"
TRAIN_LABELS = DATASET_DIR / "labels" / "train"

NAMES = {0: "swimmer", 1: "boat", 2: "jetski", 3: "life_saving_appliances", 4: "buoy"}
TARGET = 3                       # life_saving_appliances
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


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


def load_names():
    cfg = yaml.safe_load(DATASET_YAML.read_text(encoding="utf-8"))
    if cfg.get("names") != NAMES:
        raise SystemExit("class mapping mismatch: %s" % cfg.get("names"))
    return NAMES


# ---------------------------------------------------------------------------
# Step 1: overview (counts + size stats + validation)
# ---------------------------------------------------------------------------
def overview():
    names = load_names()
    for split, lbl_dir in (("train", TRAIN_LABELS), ("val", VAL_LABELS)):
        counts = Counter()
        malformed = 0
        dup_rows = 0
        for lbl in list_labels(lbl_dir):
            seen = set()
            for line in lbl.read_text(encoding="utf-8").splitlines():
                p = line.split()
                if len(p) != 5:
                    malformed += 1
                    continue
                cls = int(float(p[0])); cx, cy, w, h = map(float, p[1:5])
                if (cls not in names or not all(math.isfinite(v) for v in (cx, cy, w, h))
                        or not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1)):
                    malformed += 1
                    continue
                counts[cls] += 1
                key = tuple(round(float(x), 8) for x in p)
                if key in seen:
                    dup_rows += 1
                seen.add(key)
        print("=== %s ===" % split)
        print("  objects per class:", dict(counts))
        print("  malformed rows:", malformed, "| duplicate rows:", dup_rows)

    print("\n=== Box size (median, normalized) ===")
    for split, lbl_dir in (("train", TRAIN_LABELS), ("val", VAL_LABELS)):
        stats = {c: [] for c in names}
        for lbl in list_labels(lbl_dir):
            for (cls, cx, cy, w, h) in read_boxes(lbl):
                stats[cls].append((w, h, w * h))
        print(split.upper())
        for c in names:
            a = np.array(stats[c])
            if len(a) == 0:
                continue
            print("  %-26s n=%6d med_w=%.4f med_h=%.4f med_area=%.6f" % (
                names[c], len(a), np.median(a[:, 0]), np.median(a[:, 1]), np.median(a[:, 2])))


# ---------------------------------------------------------------------------
# Step 2: evaluate model (correct per-class metric mapping)
# ---------------------------------------------------------------------------
def evaluate():
    from ultralytics import YOLO
    names = load_names()
    if not CHECKPOINT.exists():
        raise SystemExit("checkpoint not found: " + str(CHECKPOINT))

    model = YOLO(str(CHECKPOINT))
    metrics = model.val(data=str(DATASET_YAML), imgsz=640, device=0, iou=0.7,
                        conf=0.001, split="val", plots=False, verbose=False)

    # NOTE: metrics.box.map50 / metrics.box.map are SCALARS. The per-class arrays
    # are metrics.box.ap50 (mAP@0.5) and metrics.box.ap (mAP@0.5:0.95), aligned
    # with metrics.box.ap_class_index.
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

    print("%-26s %10s %10s %10s %10s" % ("class", "P", "R", "mAP50", "mAP50-95"))
    for i in range(len(names)):
        m = per.get(i)
        if m is None:
            print("%-26s %10s %10s %10s %10s" % (names[i], "n/a", "n/a", "n/a", "n/a"))
        else:
            print("%-26s %10.4f %10.4f %10.4f %10.4f" % (
                names[i], m["p"], m["r"], m["map50"], m["map5095"]))


# ---------------------------------------------------------------------------
# Step 3: raw-vs-post diagnostic (class-3 failure modes, bounded inference)
# ---------------------------------------------------------------------------
def diagnostic():
    from ultralytics import YOLO
    names = load_names()
    if not CHECKPOINT.exists():
        raise SystemExit("checkpoint not found: " + str(CHECKPOINT))
    model = YOLO(str(CHECKPOINT))

    img_map = {p.stem: p for p in list_images(VAL_IMAGES)}
    with_c3 = sorted(lbl.stem for lbl in list_labels(VAL_LABELS)
                     if any(b[0] == TARGET for b in read_boxes(lbl)))
    sample = with_c3[:20]
    print("sample size:", len(sample))

    def run(stems, conf_th, iou_th):
        out, shapes = {}, {}
        for r in model.predict(source=[str(img_map[s]) for s in stems], conf=conf_th,
                               iou=iou_th, imgsz=640, device=0, save=False, save_txt=False,
                               plots=False, verbose=False, stream=True):
            stem = Path(r.path).stem
            shapes[stem] = (int(r.orig_shape[0]), int(r.orig_shape[1]))
            boxes = []
            if r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy()
                conf = r.boxes.conf.cpu().numpy()
                cls = r.boxes.cls.cpu().numpy().astype(int)
                for i in range(len(conf)):
                    boxes.append((int(cls[i]), float(conf[i]), tuple(xyxy[i])))
            out[stem] = boxes
        return out, shapes

    def to_xyxy(cx, cy, w, h, iw, ih):
        return (cx - w/2) * iw, (cy - h/2) * ih, (cx + w/2) * iw, (cy + h/2) * ih

    def iou(a, b):
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
        inter = iw * ih
        aa = (a[2]-a[0]) * (a[3]-a[1]); bb = (b[2]-b[0]) * (b[3]-b[1])
        return inter / (aa + bb - inter + 1e-9)

    def best(boxes, gt, keep_cls=None, skip_cls=None):
        bi, bv = -1, 0.0
        for i, (cls, cf, bb) in enumerate(boxes):
            if keep_cls is not None and cls != keep_cls:
                continue
            if skip_cls is not None and cls == skip_cls:
                continue
            v = iou(bb, gt)
            if v > bv:
                bv, bi = v, i
        return bi, bv

    raw, shapes = run(sample, 0.0001, 0.999)
    post, _ = run(sample, 0.25, 0.6)

    cat = Counter()
    for stem in sample:
        h, w = shapes[stem]
        raw_b = raw.get(stem, []); post_b = post.get(stem, [])
        for (cls, cx, cy, bw, bh) in read_boxes(VAL_LABELS / (stem + ".txt")):
            if cls != TARGET:
                continue
            gt = to_xyxy(cx, cy, bw, bh, w, h)
            ci, cv = best(raw_b, gt, keep_cls=TARGET)
            oi, ov = best(raw_b, gt, skip_cls=TARGET)
            pi, pv = best(post_b, gt, skip_cls=TARGET)
            if cv >= 0.5:
                c3_conf = raw_b[ci][1]
                survived = any(iou(pb, gt) >= 0.5 for (pc, pcf, pb) in post_b if pc == TARGET)
                if survived:
                    cat["detected"] += 1
                elif c3_conf < 0.25:
                    cat["1. weak class-3 score"] += 1
                else:
                    cat["3. same-class NMS suppression"] += 1
            elif pv >= 0.5:
                cat["2. real misclassification as " + names[post_b[pi][0]]] += 1
            elif ov >= 0.5:
                cat["raw-only overlap"] += 1
            else:
                cat["4. localization/scale miss"] += 1

    print("\nFailure-mode breakdown (class-3 GT boxes in sample):")
    for k in sorted(cat):
        print("  %-40s %d" % (k, cat[k]))
    print("Total:", sum(cat.values()))


def main():
    parser = argparse.ArgumentParser(description="Dataset inspection and model-evaluation diagnostics.")
    parser.add_argument("--step", choices=["overview", "evaluate", "diagnostic", "all"],
                        default="overview")
    args = parser.parse_args()

    if args.step in ("overview", "all"):
        overview()
    if args.step in ("evaluate", "all"):
        evaluate()
    if args.step in ("diagnostic", "all"):
        diagnostic()


if __name__ == "__main__":
    main()
