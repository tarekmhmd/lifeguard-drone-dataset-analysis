"""Zero-shot comparison of pose models on HAND-CORRECTED frames (CVAT export, COCO keypoints 1.0).

Metrics (all on GT keypoints with visibility > 0; normaliser = longer side of the GT box):
  PCK@0.05 / 0.1 / 0.2 on upper body (nose, shoulders, elbows, wrists) and on all 17,
  share of frames with >= 3 confident (>=0.5) upper-body keypoints, latency per person.
Models (any subset):
  --vitpose-onnx   ViTPose-L SwimXYZ ONNX, top-down on the GT box
  --yolo-pose      Ultralytics pose weights, full frame; the prediction with best IoU to the GT box is used
Only run on the `real_val` frames; select with --split-manifest + --split real_val. Report per camera type via --group.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from pose_common import UPPER_BODY, OnnxVitPose, confident_upper, iou, pck

THRS = (0.05, 0.1, 0.2)
FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]  # COCO-17 left <-> right


def load_gt(coco_path: Path) -> dict[str, tuple[np.ndarray, tuple[float, float, float, float]]]:
    d = json.loads(coco_path.read_text(encoding="utf-8"))
    names = {i["id"]: i["file_name"] for i in d["images"]}
    out = {}
    for a in d["annotations"]:
        k = np.array(a["keypoints"], dtype=np.float32).reshape(-1, 3)
        if k.shape[0] != 17:
            raise SystemExit(f"annotation {a['id']} has {k.shape[0]} keypoints, expected 17 (COCO order)")
        x, y, w, h = a["bbox"]
        out[Path(names[a["image_id"]]).name] = (k, (x, y, x + w, y + h))
    return out


def run_vitpose(model: OnnxVitPose, img: np.ndarray, box: tuple[float, ...]) -> np.ndarray:
    return model.predict(img, [box])[0]


def run_yolo(model, img: np.ndarray, box: tuple[float, ...]) -> np.ndarray | None:  # noqa: ANN001
    res = model.predict(img, verbose=False)[0]
    if res.keypoints is None or len(res.boxes) == 0:
        return None
    best, best_iou = None, 0.0
    for i, b in enumerate(res.boxes.xyxy.cpu().numpy()):
        v = iou(tuple(b), box)
        if v > best_iou:
            best, best_iou = i, v
    if best is None or best_iou < 0.3:
        return None
    xy = res.keypoints.xy[best].cpu().numpy()
    cf = res.keypoints.conf[best].cpu().numpy() if res.keypoints.conf is not None else np.ones(17)
    return np.concatenate([xy, cf[:, None]], axis=1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt", type=Path, required=True, help="CVAT export (COCO keypoints 1.0) json")
    ap.add_argument("--images", type=Path, required=True, help="folder with the images of the set")
    ap.add_argument("--manifest", type=Path, help="manifest.csv, to restrict to --split")
    ap.add_argument("--split", default="real_val")
    ap.add_argument("--group", default="", help="substring of image name defining a camera group, e.g. 'ddt'")
    ap.add_argument("--vitpose-onnx", type=str)
    ap.add_argument("--yolo-pose", type=str)
    ap.add_argument("--out", type=Path, default=Path("zero_shot_results.csv"))
    ap.add_argument("--boxes-from-manifest", action="store_true",
                    help="use the manifest person boxes (x1..y2) instead of the GT bbox. Use for CVAT exports, whose "
                         "bbox is fitted tightly around the annotated points")
    a = ap.parse_args()

    if not a.gt.is_file():
        raise SystemExit(f"GT file not found: {a.gt}. Export the corrected real_val frames from CVAT first "
                         "(see docs/POSE_ANNOTATION_GUIDE.md); this step cannot run before that.")
    gt = load_gt(a.gt)
    if a.boxes_from_manifest:
        if not a.manifest:
            raise SystemExit("--boxes-from-manifest needs --manifest")
        mboxes = {Path(r["image"]).name: tuple(float(r[k]) for k in ("x1", "y1", "x2", "y2"))
                  for r in csv.DictReader(open(a.manifest, encoding="utf-8"))}
        gt = {n: (k, mboxes[n]) for n, (k, _) in gt.items() if n in mboxes}
    names = set(gt)
    if a.manifest:
        names &= {Path(r["image"]).name for r in csv.DictReader(open(a.manifest, encoding="utf-8"))
                  if r["split"] == a.split}
    if a.group:
        names = {n for n in names if a.group in n}
    names = sorted(names)
    if not names:
        raise SystemExit("no frames selected: check --split / --group / file names")
    print(f"{len(names)} frames, split={a.split}, group={a.group or 'all'}")

    runners = {}
    if a.vitpose_onnx:
        m = OnnxVitPose(a.vitpose_onnx)
        runners["vitpose_swimxyz"] = lambda im, b: run_vitpose(m, im, b)
    if a.yolo_pose:
        from ultralytics import YOLO

        y = YOLO(a.yolo_pose)
        runners[Path(a.yolo_pose).stem] = lambda im, b: run_yolo(y, im, b)
    if not runners:
        raise SystemExit("give --vitpose-onnx and/or --yolo-pose")

    rows = []
    for mname, fn in runners.items():
        acc = {(s, t): [0, 0] for s in ("upper", "all") for t in THRS}
        acc_sw = {(s, t): [0, 0] for s in ("upper", "all") for t in THRS}
        n_conf3 = n_missing = n_swapped = 0
        times = []
        for n in names:
            img = cv2.imread(str(a.images / n))
            k, box = gt[n]
            t0 = time.perf_counter()
            pred = fn(img, box)
            times.append((time.perf_counter() - t0) * 1000)
            if pred is None:
                n_missing += 1
                pred = np.zeros((17, 3), dtype=np.float32)
                pred[:, :2] = -1e6
            n_conf3 += confident_upper(pred) >= 3
            # swap-tolerant: one left/right decision per frame (whole skeleton mirrored in labels, positions kept),
            # chosen by the larger PCK@0.2 over all keypoints; ties keep the original labelling
            swapped = pred[FLIP_IDX]
            use_swap = pck(swapped, k, box, 0.2)[0] > pck(pred, k, box, 0.2)[0]
            n_swapped += use_swap
            best = swapped if use_swap else pred
            for t in THRS:
                for s, idx in (("upper", UPPER_BODY), ("all", None)):
                    ok, tot = pck(pred, k, box, t, idx)
                    acc[(s, t)][0] += ok
                    acc[(s, t)][1] += tot
                    ok2, _ = pck(best, k, box, t, idx)
                    acc_sw[(s, t)][0] += ok2
                    acc_sw[(s, t)][1] += tot
        row = {"model": mname, "frames": len(names), "no_prediction": n_missing,
               "frames_ge3_confident_upper": round(n_conf3 / len(names), 3),
               "latency_ms_median_per_person": round(float(np.median(times)), 1),
               "frames_better_with_lr_swap": n_swapped}
        for (s, t), (ok, tot) in acc.items():
            row[f"PCK@{t}_{s}"] = round(ok / tot, 3) if tot else None
        for (s, t), (ok, tot) in acc_sw.items():
            row[f"PCK@{t}_{s}_swaptol"] = round(ok / tot, 3) if tot else None
        rows.append(row)
        print(row)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
