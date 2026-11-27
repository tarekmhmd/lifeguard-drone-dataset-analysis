"""Does a pose model's keypoint confidence carry information? (E2 check; agreement with reference labels, not accuracy)

For every labelled reference keypoint, a point is a HIT if the model's point (after one left/right decision per frame,
as in the swap-tolerant PCK) is within --thr of the box's longer side. The script reports the mean confidence of hits and
misses and the AUROC of confidence for separating them. AUROC 0.5 = confidence says nothing (E1b's always-high
confidence); higher is better. Frames without a matched detection count as misses with confidence 0.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from pose_common import iou
from zero_shot_eval import load_gt

FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUROC (ties get average rank); nan if only one class is present."""
    pos, neg = labels.sum(), (~labels).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    order = scores.argsort(kind="mergesort")
    ranks = np.empty(len(scores))
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[labels].sum() - pos * (pos + 1) / 2) / (pos * neg))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", type=Path, required=True, help="COCO keypoint reference JSON")
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, help="restrict to --split (and optionally use its boxes)")
    ap.add_argument("--split")
    ap.add_argument("--boxes-from-manifest", action="store_true")
    ap.add_argument("--weights", nargs="+", required=True, help="name=path.pt entries")
    ap.add_argument("--thr", type=float, default=0.1)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    from ultralytics import YOLO

    gt = load_gt(a.gt)
    names = set(gt)
    rows = list(csv.DictReader(open(a.manifest, encoding="utf-8"))) if a.manifest else []
    if a.manifest:
        names &= {Path(r["image"]).name for r in rows if not a.split or r["split"] == a.split}
    if a.boxes_from_manifest:
        mb = {Path(r["image"]).name: tuple(float(r[k]) for k in ("x1", "y1", "x2", "y2")) for r in rows}
        gt = {n: (k, mb[n]) for n, (k, _) in gt.items() if n in mb}
    names = sorted(n for n in names if n in gt)

    results = []
    for spec in a.weights:
        tag, w = spec.split("=", 1)
        m = YOLO(w)
        confs, hits = [], []
        for n in names:
            ref, box = gt[n]
            lab = ref[:, 2] > 0
            size = max(box[2] - box[0], box[3] - box[1], 1.0)
            r = m.predict(cv2.imread(str(a.images / n)), verbose=False)[0]
            pred = None
            if r.keypoints is not None and len(r.boxes):
                ious = [iou(tuple(b), box) for b in r.boxes.xyxy.cpu().numpy()]
                i = int(np.argmax(ious))
                if ious[i] >= 0.3:
                    pred = np.concatenate([r.keypoints.xy[i].cpu().numpy(), r.keypoints.conf[i].cpu().numpy()[:, None]], 1)
            if pred is None:
                confs += [0.0] * int(lab.sum()); hits += [False] * int(lab.sum())
                continue
            def hit_of(p):
                return np.hypot(*(p[:, :2] - ref[:, :2]).T) / size <= a.thr
            h0, h1 = hit_of(pred), hit_of(pred[FLIP_IDX])
            if h1[lab].sum() > h0[lab].sum():
                pred, h0 = pred[FLIP_IDX], h1
            confs += pred[lab, 2].tolist(); hits += h0[lab].tolist()
        c, h = np.array(confs), np.array(hits, dtype=bool)
        res = {"model": tag, "frames": len(names), "points": len(c), "hit_rate": round(float(h.mean()), 3),
               "mean_conf_hits": round(float(c[h].mean()), 3) if h.any() else float("nan"),
               "mean_conf_misses": round(float(c[~h].mean()), 3) if (~h).any() else float("nan"),
               "conf_auroc": round(auroc(c, h), 3)}
        results.append(res)
        print(res)
    if a.out:
        with open(a.out, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(results[0])); wr.writeheader(); wr.writerows(results)
        print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
