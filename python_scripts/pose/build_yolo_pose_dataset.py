"""Build an Ultralytics YOLO-pose dataset from AI reference labels (pseudo-labels, provenance ai_reference_unverified).

Splits (no frame in two splits):
  train = manifest real_train (300)          -> pseudo-labels to learn from
  val   = manifest real_val NOT in test list  -> model selection / early stopping
  test  = the frames in --test-list (27)      -> exploratory evaluation only, never trained on
Boxes come from the manifest (the source dataset's own person boxes). Keypoints come from --refs; a keypoint is
labelled (v=2) only if its ViTPose score >= --min-score, else (0, 0, 0) so the loss ignores it.
Images are copied; nothing in --set is modified.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]  # COCO-17 left/right swap


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", type=Path, required=True, help="folder with manifest.csv and images/")
    ap.add_argument("--refs", type=Path, required=True, help="AI reference JSON for all frames (make_ai_reference.py)")
    ap.add_argument("--test-list", type=Path, required=True, help="text file: test frame names, one per line")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-score", type=float, default=0.5)
    a = ap.parse_args()

    rows = {Path(r["image"]).name: r for r in csv.DictReader(open(a.set / "manifest.csv", encoding="utf-8"))}
    test = {ln.strip() for ln in a.test_list.read_text().splitlines() if ln.strip()}
    bad = [n for n in test if n not in rows or rows[n]["split"] != "real_val"]
    if bad:
        raise SystemExit(f"test frames must be real_val frames in the manifest: {bad}")
    refs = json.loads(a.refs.read_text(encoding="utf-8"))
    img_by_id = {i["id"]: i for i in refs["images"]}
    ann_by_name = {img_by_id[x["image_id"]]["file_name"]: x for x in refs["annotations"]}
    missing = sorted(set(rows) - set(ann_by_name))
    if missing:
        raise SystemExit(f"{len(missing)} manifest frames have no reference annotation, e.g. {missing[:3]}")

    def split_of(n: str) -> str:
        if n in test:
            return "test"
        return "train" if rows[n]["split"] == "real_train" else "val"

    for sp in ("train", "val", "test"):
        (a.out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (a.out / "labels" / sp).mkdir(parents=True, exist_ok=True)
    counts, vis = Counter(), Counter()
    with open(a.out / "split_manifest.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["image", "split", "label_provenance"])
        for n, r in sorted(rows.items()):
            sp = split_of(n)
            ann = ann_by_name[n]
            im = img_by_id[ann["image_id"]]
            w, h = im["width"], im["height"]
            x1, y1, x2, y2 = (float(r[k]) for k in ("x1", "y1", "x2", "y2"))
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            vals = [0, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h]
            kp = ann["keypoints"]
            scores = ann.get("scores") or [1.0 if kp[3 * k + 2] > 0 else 0.0 for k in range(17)]
            for k in range(17):
                x, y, v = kp[3 * k: 3 * k + 3]
                if v > 0 and scores[k] >= a.min_score:
                    vals += [min(max(x / w, 0), 1), min(max(y / h, 0), 1), 2]
                    vis[sp] += 1
                else:
                    vals += [0, 0, 0]
            line = " ".join(str(int(v)) if i == 0 or (i >= 5 and (i - 5) % 3 == 2) else f"{v:.6f}"
                            for i, v in enumerate(vals))
            (a.out / "labels" / sp / (Path(n).stem + ".txt")).write_text(line + "\n")
            shutil.copyfile(a.set / r["image"], a.out / "images" / sp / n)
            counts[sp] += 1
            wr.writerow([n, sp, "ai_reference_unverified"])

    yaml = (f"path: {a.out.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
            f"kpt_shape: [17, 3]\nflip_idx: {FLIP_IDX}\nnames:\n  0: person\n"
            "# labels: ViTPose-L SwimXYZ pseudo-labels (ai_reference_unverified), NOT human ground truth\n")
    (a.out / "data.yaml").write_text(yaml, encoding="utf-8")

    # validation of what was written
    for sp in ("train", "val", "test"):
        for lf in (a.out / "labels" / sp).glob("*.txt"):
            t = lf.read_text().split()
            assert len(t) == 56, f"{lf}: {len(t)} tokens, expected 56"
            assert all(0.0 <= float(v) <= 1.0 for v in t[1:5]), f"{lf}: box out of range"
            assert all(float(v) in (0.0, 2.0) for v in t[7::3]), f"{lf}: bad visibility"
    names = {sp: {p.name for p in (a.out / "images" / sp).iterdir()} for sp in ("train", "val", "test")}
    assert not (names["train"] & names["val"] or names["train"] & names["test"] or names["val"] & names["test"])
    print(f"dataset at {a.out}: {dict(counts)} frames (no overlap between splits)")
    for sp in ("train", "val", "test"):
        print(f"  {sp:5s} labelled keypoints per frame: {vis[sp] / max(counts[sp], 1):.1f} of 17")
    print(f"min_score={a.min_score}; labels are ai_reference_unverified pseudo-labels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
