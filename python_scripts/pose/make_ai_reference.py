"""ViTPose-L SwimXYZ keypoints (with per-keypoint scores) for chosen frames, saved as COCO keypoints.

Every annotation is marked provenance = "ai_reference_unverified". These are MODEL OUTPUTS, not human ground truth:
  - as training labels they are pseudo-labels (teacher -> student distillation);
  - as evaluation targets they measure agreement with ViTPose, not accuracy.
Keypoints with score < --min-score are written unlabelled (0, 0, 0); the raw scores are kept in "scores".
Inputs are only read: images, manifest.csv and existing JSON files are never modified.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from pose_common import COCO17, COCO_SKELETON, UPPER_BODY, OnnxVitPose

PROVENANCE = "ai_reference_unverified"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", type=Path, required=True, help="folder with manifest.csv and images/")
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", type=Path, required=True, help="output JSON path")
    ap.add_argument("--frames", type=Path, help="optional text file, one image file name per line")
    ap.add_argument("--min-score", type=float, default=0.5)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.set / "manifest.csv", encoding="utf-8")))
    if a.frames:
        want = {ln.strip() for ln in a.frames.read_text().splitlines() if ln.strip()}
        rows = [r for r in rows if Path(r["image"]).name in want]
        missing = want - {Path(r["image"]).name for r in rows}
        if missing:
            raise SystemExit(f"frames not in manifest: {sorted(missing)}")
    model = OnnxVitPose(a.onnx)
    images, anns, all_scores = [], [], []
    for i, r in enumerate(sorted(rows, key=lambda r: r["image"]), start=1):
        img = cv2.imread(str(a.set / r["image"]))
        if img is None:
            raise SystemExit(f"cannot read {r['image']}")
        h, w = img.shape[:2]
        box = tuple(float(r[k]) for k in ("x1", "y1", "x2", "y2"))
        kp = model.predict(img, [box])[0]
        flat, n_lab = [], 0
        for x, y, s in kp:
            if s >= a.min_score and 0 <= x <= w and 0 <= y <= h:
                flat += [round(float(x), 1), round(float(y), 1), 2]
                n_lab += 1
            else:
                flat += [0, 0, 0]
        all_scores.append(kp[:, 2])
        images.append({"id": i, "file_name": Path(r["image"]).name, "width": w, "height": h})
        anns.append({"id": i, "image_id": i, "category_id": 1, "iscrowd": 0, "num_keypoints": n_lab,
                     "keypoints": flat, "scores": [round(float(s), 4) for s in kp[:, 2]],
                     "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                     "area": (box[2] - box[0]) * (box[3] - box[1]),
                     "split": r["split"], "provenance": PROVENANCE})
    coco = {"images": images, "annotations": anns,
            "categories": [{"id": 1, "name": "person", "supercategory": "person",
                            "keypoints": COCO17, "skeleton": COCO_SKELETON}],
            "info": {"provenance": PROVENANCE, "model": Path(a.onnx).name, "min_score": a.min_score,
                     "note": "Model predictions. NOT human ground truth."}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(coco), encoding="utf-8")

    sc = np.array(all_scores)
    print(f"wrote {a.out}: {len(images)} frames, provenance={PROVENANCE}, min_score={a.min_score}")
    print(f"labelled keypoints per frame: mean {np.mean([x['num_keypoints'] for x in anns]):.1f} of 17")
    print(f"frames with < 3 upper-body keypoints labelled: "
          f"{int(sum((s[UPPER_BODY] >= a.min_score).sum() < 3 for s in sc))}")
    print("keypoint            mean score   missing (< min_score)")
    for k, name in enumerate(COCO17):
        print(f"  {name:16s}  {sc[:, k].mean():.2f}        {int((sc[:, k] < a.min_score).sum())}/{len(sc)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
