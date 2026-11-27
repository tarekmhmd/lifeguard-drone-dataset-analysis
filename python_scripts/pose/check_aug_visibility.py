"""Check what Ultralytics training augmentation does to keypoint visibility flags (run BEFORE training).

Background (E1 v1): unknown points were written as (0, 0, v=1). Mosaic/affine augmentation sets v=0 for every point that
lands outside the image, so most "unknown" points reached the loss as "absent" and the model learned low confidence.
This script loads training samples exactly as the trainer does (mosaic, affine, flip) and compares the share of
v=0 / v=1 / v=2 in the label files with the share after augmentation.
Pass: the ratio v=1 / v=2 stays about the same (mosaic also crops real points away; that part is normal).
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np


def vbin(v: float) -> int:
    """0 absent, 1 unknown (v=1 or soft 1 <= v < 2), 2 confident."""
    return 0 if v < 0.5 else (2 if v > 1.995 else 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="data.yaml of the pose dataset")
    ap.add_argument("--n", type=int, default=300, help="training samples to draw")
    ap.add_argument("--imgsz", type=int, default=960)
    a = ap.parse_args()

    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset

    data = check_det_dataset(str(a.data))
    cfg = get_cfg(overrides=dict(task="pose", imgsz=a.imgsz, fliplr=0.5, flipud=0.0, mosaic=1.0))
    ds = build_yolo_dataset(cfg, data["train"], 16, data, mode="train")

    before = Counter()
    for lf in Path(data["train"].replace("images", "labels")).glob("*.txt"):
        for ln in lf.read_text().splitlines():
            before.update(vbin(float(v)) for v in ln.split()[7::3])
    after = Counter()
    rng = np.random.default_rng(0)
    for i in rng.choice(len(ds), size=min(a.n, len(ds)), replace=False):
        k = ds[int(i)]["keypoints"]
        after.update(vbin(float(v)) for v in k[..., 2].reshape(-1).tolist())

    def share(c: Counter) -> str:
        t = max(sum(c.values()), 1)
        return "  ".join(f"v={v}: {c[v] / t:.2f}" for v in (0, 1, 2))

    print("label files:        ", share(before))
    print("after augmentation: ", share(after))
    # mosaic crops real points away too (v=2 -> 0 is normal); unknown points must be lost no faster than known ones
    rb = before[1] / max(before[2], 1)
    ra = after[1] / max(after[2], 1)
    print(f"unknown/known ratio: label files {rb:.2f}, after augmentation {ra:.2f}")
    ok = rb == 0 or ra >= 0.8 * rb
    print("PASS: unknown points survive augmentation" if ok else
          "FAIL: augmentation turns most unknown (v=1) points into absent (v=0) - do not train")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
