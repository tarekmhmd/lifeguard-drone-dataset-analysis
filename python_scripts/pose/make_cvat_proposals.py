"""Write model keypoint PROPOSALS as a COCO-keypoints JSON for import into CVAT (to be corrected by hand).

Input: manifest.csv from select_annotation_frames.py. Proposals are NOT ground truth: provenance is
`model_proposal_unverified` until a human has corrected the frame in CVAT.
Keypoints with score < --min-score are written as unlabelled (0, 0, 0) so the annotator adds them only if visible.

--blank-splits (default real_val): frames of these splits get NO model keypoints (box only, all points unlabelled),
so ground truth is annotated from scratch and is not biased toward the proposal model. Their provenance is `human`.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2

from pose_common import COCO17, COCO_SKELETON, OnnxVitPose


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", type=Path, required=True, help="folder with manifest.csv and images/")
    ap.add_argument("--onnx", type=str, required=True, help="path to the pose .onnx file")
    ap.add_argument("--min-score", type=float, default=0.3)
    ap.add_argument("--blank-splits", nargs="*", default=["real_val"],
                    help="splits written without proposals (annotate from scratch); empty list = proposals everywhere")
    a = ap.parse_args()
    model = None  # built lazily: not needed if every split is blank
    rows = list(csv.DictReader(open(a.set / "manifest.csv", encoding="utf-8")))
    images, anns = [], []
    for i, r in enumerate(rows, start=1):
        img = cv2.imread(str(a.set / r["image"]))
        if img is None:
            raise SystemExit(f"cannot read {r['image']}")
        h, w = img.shape[:2]
        box = tuple(float(r[k]) for k in ("x1", "y1", "x2", "y2"))
        flat: list[float] = []
        n_lab = 0
        if r["split"] in a.blank_splits:
            flat = [0, 0, 0] * len(COCO17)
        else:
            if model is None:
                model = OnnxVitPose(a.onnx)
            for x, y, s in model.predict(img, [box])[0]:
                if s >= a.min_score and 0 <= x <= w and 0 <= y <= h:
                    flat += [round(float(x), 1), round(float(y), 1), 2]
                    n_lab += 1
                else:
                    flat += [0, 0, 0]
        images.append({"id": i, "file_name": Path(r["image"]).name, "width": w, "height": h})
        anns.append({"id": i, "image_id": i, "category_id": 1, "iscrowd": 0, "num_keypoints": n_lab,
                     "keypoints": flat, "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                     "area": (box[2] - box[0]) * (box[3] - box[1])})
    coco = {"images": images, "annotations": anns,
            "categories": [{"id": 1, "name": "person", "supercategory": "person",
                            "keypoints": COCO17, "skeleton": COCO_SKELETON}],
            "info": {"description": "model proposals, provenance=model_proposal_unverified"}}
    out = a.set / "proposals_coco_keypoints.json"
    out.write_text(json.dumps(coco), encoding="utf-8")
    n_blank = sum(1 for r in rows if r["split"] in a.blank_splits)
    print(f"wrote {out} ({len(images)} images; {n_blank} blank for scratch annotation, "
          f"{len(images) - n_blank} with model proposals)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
