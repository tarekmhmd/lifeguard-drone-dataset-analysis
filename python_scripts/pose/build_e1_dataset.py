"""E1: YOLO-pose dataset from filtered ViTPose teacher labels (offline teacher only; ViTPose is never deployed).

Source: E0 predictions (`e0_predictions.jsonl`: ViTPose on every frame and on its mirrored copy) plus the dataset's own
person boxes. A ViTPose keypoint becomes a training label (v=2) only if
  confidence >= --conf  AND  flip error <= --flip-max  AND  (no previous frame, or jump <= --jump-max)
(distances as fractions of the box's longer side). Every other keypoint is UNKNOWN:
  train labels: v=1 (ignored by the patched loss, see ignore_aware_pose_loss.py), placed at the box centre so that
                augmentation does not push it outside the image (which would make Ultralytics set v=0)
  val labels:   v=0 (Ultralytics' OKS metric treats v>0 as labelled)
  --soft-conf (E2): unknown train points get v = 1 + min(teacher confidence, 0.99); see soft_conf_pose_loss.py
Splits: held-out blocks (2, 13, 15, 16; the E3/test frames come from them) are never used.
Whole frame-index blocks (about --val-frac of the frames) become the validation split; no neighbouring frames are
shared between train and validation.
Provenance of every label: vitpose_teacher_filtered (pseudo-label, not ground truth).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
COCO17 = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow",
          "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle",
          "right_ankle"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="dataset root with images/{train,val}, labels/{train,val}")
    ap.add_argument("--pred", type=Path, required=True, help="E0 predictions JSONL")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--conf", type=float, default=0.7)
    ap.add_argument("--flip-max", type=float, default=0.05)
    ap.add_argument("--jump-max", type=float, default=0.25)
    ap.add_argument("--min-points", type=int, default=3, help="drop frames with fewer confident points")
    ap.add_argument("--val-frac", type=float, default=0.15,
                    help="target share of non-held-out frames used for validation (whole blocks)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--soft-conf", action="store_true",
                    help="E2: unknown train points get v = 1 + teacher confidence (needs soft_conf_pose_loss)")
    a = ap.parse_args()

    recs = sorted((json.loads(ln) for ln in a.pred.read_text().splitlines() if ln.strip()), key=lambda r: r["idx"])
    by_idx = {r["idx"]: r for r in recs}
    pool = [r for r in recs if not r["heldout"]]
    counts = Counter(r["block"] for r in pool)
    # whole blocks to validation until ~val_frac of the frames; skip blocks that alone overshoot 1.5x the target
    target = a.val_frac * len(pool)
    cands = [b for b, n in counts.items() if n >= 50 and n <= 1.5 * target]
    random.Random(a.seed).shuffle(cands)
    val_blocks, n_val = set(), 0
    for b in cands:
        if n_val >= target:
            break
        val_blocks.add(b)
        n_val += counts[b]
    if not val_blocks:
        val_blocks = {min(counts, key=counts.get)}

    for sp in ("train", "val"):
        (a.out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (a.out / "labels" / sp).mkdir(parents=True, exist_ok=True)
    kept = Counter()
    dropped = 0
    lab = {sp: np.zeros(17) for sp in ("train", "val")}
    man = open(a.out / "e1_manifest.csv", "w", newline="", encoding="utf-8")
    wr = csv.writer(man)
    wr.writerow(["image", "split", "block", "frame", "confident_points", "label_provenance"])
    for r in pool:
        sp = "val" if r["block"] in val_blocks else "train"
        w, h = r["W"], r["H"]
        x1, y1, x2, y2 = r["box"]
        size = max(x2 - x1, y2 - y1, 1.0)
        c = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        kv, kf = np.array(r["vit"]), np.array(r["vit_f"])
        prev = next((by_idx[r["idx"] - g] for g in (1, 2, 3) if r["idx"] - g in by_idx), None)
        vals, n_conf, good_idx = [0, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h], 0, []
        for i in range(17):
            x, y, s = kv[i]
            good = s >= a.conf and kf[i][2] >= a.conf and np.hypot(*(kv[i][:2] - kf[i][:2])) / size <= a.flip_max
            if good and prev is not None and prev["vit"][i][2] >= a.conf:
                pc = np.array([(prev["box"][0] + prev["box"][2]) / 2, (prev["box"][1] + prev["box"][3]) / 2])
                jump = np.hypot(*((kv[i][:2] - c) - (np.array(prev["vit"][i][:2]) - pc))) / size
                good = jump <= a.jump_max
            good = good and 0 <= x < w and 0 <= y < h
            if good:
                vals += [x / w, y / h, 2]
                n_conf += 1
                good_idx.append(i)
            else:
                # unknown: placed at the box centre, not (0, 0). Ultralytics augmentation sets v=0 for any point
                # pushed outside the image, and (0, 0) usually is, which turned "unknown" into "absent" (E1 v1).
                vu = (1 + round(min(float(s), 0.99), 3) if a.soft_conf else 1) if sp == "train" else 0
                vals += [(x1 + x2) / 2 / w, (y1 + y2) / 2 / h, vu]
        if n_conf < a.min_points:
            dropped += 1
            continue
        src = a.root / "images" / r["folder"] / r["file"]
        shutil.copyfile(src, a.out / "images" / sp / r["file"])
        line = " ".join((str(int(v)) if float(v).is_integer() else f"{v:.3f}") if i == 0 or (i >= 5 and (i - 5) % 3 == 2)
                        else f"{v:.6f}" for i, v in enumerate(vals))
        (a.out / "labels" / sp / (Path(r["file"]).stem + ".txt")).write_text(line + "\n")
        kept[sp] += 1
        lab[sp][good_idx] += 1  # coverage counted over kept frames only
        wr.writerow([r["file"], sp, r["block"], r["idx"], n_conf, "vitpose_teacher_filtered"])
    man.close()

    (a.out / "data.yaml").write_text(
        f"path: {a.out.as_posix()}\ntrain: images/train\nval: images/val\nkpt_shape: [17, 3]\n"
        f"flip_idx: {FLIP_IDX}\nnames:\n  0: person\n"
        "# train labels: v=2 confident ViTPose point, v=1 unknown (needs ignore_aware_pose_loss), v=0 absent\n"
        "# val labels: v=2 or 0. Provenance: vitpose_teacher_filtered (pseudo-labels, not ground truth)\n",
        encoding="utf-8")

    for sp in ("train", "val"):
        for lf in (a.out / "labels" / sp).glob("*.txt"):
            t = lf.read_text().split()
            assert len(t) == 56 and all(0.0 <= float(v) <= 1.0 for v in t[1:5]), lf
            vs = set(float(v) for v in t[7::3])
            if sp == "val":
                assert vs <= {0.0, 2.0}, lf
            elif a.soft_conf:
                assert all(v == 0.0 or 1.0 <= v <= 1.99 or v == 2.0 for v in vs), lf
            else:
                assert vs <= {0.0, 1.0, 2.0}, lf
    print(f"thresholds: conf >= {a.conf}, flip <= {a.flip_max}, jump <= {a.jump_max}, min points {a.min_points}"
          f"{'; unknown train points: v = 1 + teacher confidence (soft)' if a.soft_conf else ''}")
    print(f"held-out blocks excluded; validation blocks: {sorted(val_blocks)}")
    print(f"frames kept: {dict(kept)}; dropped (< {a.min_points} confident points): {dropped}")
    print("keypoint          train coverage   val coverage   (fraction of kept frames with a confident label)")
    for i, nm in enumerate(COCO17):
        print(f"  {nm:16s} {lab['train'][i] / max(kept['train'], 1):.2f}            "
              f"{lab['val'][i] / max(kept['val'], 1):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
