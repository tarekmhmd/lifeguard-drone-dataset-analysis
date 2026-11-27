"""Pick frames to annotate with keypoints, split by contiguous frame-index BLOCKS (no video ids exist).

Source: a YOLO detection dataset (images/{train,val}, labels/{train,val}), e.g. `datasets (1).zip` unzipped.
Source train/val folders are ignored (they overlap in frame index, so they leak): all frames are pooled,
cut into blocks of --block frames, whole blocks go to real_val or real_train, and frames within --buffer of a
block boundary that touches a block of the OTHER split are dropped. Class labels are NOT used or exported.

Writes into --out: images/ (copies), manifest.csv (image,split,block,frame,x1,y1,x2,y2).
"""
from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
from pathlib import Path


def frame_index(p: Path) -> int | None:
    m = re.search(r"(\d+)$", p.stem)
    return int(m.group(1)) if m else None


def read_box(label: Path, w: int, h: int) -> tuple[float, float, float, float] | None:
    boxes = []
    for ln in label.read_text(errors="replace").splitlines():
        t = ln.split()
        if len(t) == 5:
            cx, cy, bw, bh = (float(v) for v in t[1:])
            boxes.append(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
    return boxes[0] if len(boxes) == 1 else None  # skip images with 0 or >1 boxes (unambiguous only)


def select(root: Path, n_val: int, n_train: int, block: int, buffer: int, spacing: int, seed: int
           ) -> list[dict]:
    items = []
    for sp in ("train", "val"):
        for img in sorted((root / "images" / sp).glob("*.jpg")):
            idx = frame_index(img)
            lbl = root / "labels" / sp / (img.stem + ".txt")
            if idx is not None and lbl.is_file():
                items.append((idx, img, lbl))
    items.sort(key=lambda t: t[0])
    count: dict[int, int] = {}
    for idx, _, _ in items:
        count[idx // block] = count.get(idx // block, 0) + 1
    blocks = sorted(count)
    rng = random.Random(seed)
    order = blocks[:]
    rng.shuffle(order)
    # Give blocks to real_val until they hold enough frames (blocks are very uneven in frame count),
    # then everything else is real_train. Frames still thin out via --spacing and the edge buffer.
    split_of: dict[int, str] = {}
    have = 0
    for b in order:
        if have < 2 * n_val:
            split_of[b] = "real_val"
            have += count[b]
        else:
            split_of[b] = "real_train"

    def near_other_split(idx: int) -> bool:
        b = idx // block
        for nb, edge in ((b - 1, idx - b * block < buffer), (b + 1, (b + 1) * block - idx <= buffer)):
            if edge and nb in split_of and split_of[nb] != split_of[b]:
                return True
        return False

    pool = [(i, im, lb) for i, im, lb in items if not near_other_split(i)]
    rng.shuffle(pool)
    chosen: list[dict] = []
    per_split = {"real_val": 0, "real_train": 0}
    target = {"real_val": n_val, "real_train": n_train}
    taken: dict[str, list[int]] = {"real_val": [], "real_train": []}
    from PIL import Image

    for idx, img, lbl in pool:
        sp = split_of[idx // block]
        if per_split[sp] >= target[sp] or any(abs(idx - t) < spacing for t in taken[sp]):
            continue
        with Image.open(img) as im:
            w, h = im.size
        box = read_box(lbl, w, h)
        if box is None:
            continue
        taken[sp].append(idx)
        per_split[sp] += 1
        chosen.append({"src": img, "split": sp, "block": idx // block, "frame": idx, "box": box})
    return chosen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-val", type=int, default=150)
    ap.add_argument("--n-train", type=int, default=300)
    ap.add_argument("--block", type=int, default=3000, help="frames per block (unit: frame index)")
    ap.add_argument("--buffer", type=int, default=300, help="drop frames this close to a cross-split block edge")
    ap.add_argument("--spacing", type=int, default=12, help="min frame-index distance between chosen frames")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    chosen = select(a.root, a.n_val, a.n_train, a.block, a.buffer, a.spacing, a.seed)
    (a.out / "images").mkdir(parents=True, exist_ok=True)
    with open(a.out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["image", "split", "block", "frame", "x1", "y1", "x2", "y2"])
        for c in chosen:
            dst = a.out / "images" / c["src"].name
            shutil.copyfile(c["src"], dst)
            wr.writerow([f"images/{dst.name}", c["split"], c["block"], c["frame"],
                         *[round(v, 1) for v in c["box"]]])
    cnt = {s: sum(1 for c in chosen if c["split"] == s) for s in ("real_val", "real_train")}
    blocks = {s: sorted({c["block"] for c in chosen if c["split"] == s}) for s in cnt}
    print("selected:", cnt, "(requested", a.n_val, "/", a.n_train, ")")
    for sp, want in (("real_val", a.n_val), ("real_train", a.n_train)):
        if cnt[sp] < want:
            print(f"WARNING: only {cnt[sp]} {sp} frames available; lower --spacing/--buffer or --block")
    print("blocks per split (disjoint by construction):", {s: len(b) for s, b in blocks.items()})
    assert not set(blocks["real_val"]) & set(blocks["real_train"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
