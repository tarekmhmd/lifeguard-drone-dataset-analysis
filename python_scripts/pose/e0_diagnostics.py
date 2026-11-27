"""E0: label-free pose diagnostics (inference only, no training, no labels).

Two modes:
  --run      ViTPose-L SwimXYZ (top-down, on the dataset box) and YOLO11s-pose (full frame, matched to the box) on
             every frame and on its horizontally flipped copy. Appends one JSON line per frame (resumable).
  --analyze  Per-keypoint coverage, flip consistency, cross-model agreement, temporal jitter, confidence-vs-agreement,
             consensus coverage (E1 feasibility) and the most disputed held-out frames (E3 candidates).
All numbers are consistency / agreement measures. None of them is accuracy: there is no ground truth here.

Distances are normalised by the longer side of the dataset box. "ok" means confidence >= --conf.
Held-out = frame-index blocks 2, 13, 15, 16 (block size 3000), the real_val blocks of the pose set: never train on them.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

from pose_common import COCO17, UPPER_BODY

FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
LOWER_BODY = [11, 12, 13, 14, 15, 16]
HELDOUT_BLOCKS = {2, 13, 15, 16}
BLOCK = 3000
# Proposed thresholds (fractions of box size); documented in the report, not tuned on any labels.
CROSS_OK, FLIP_OK, JUMP_OK, DISPUTE = 0.10, 0.10, 0.25, 0.20


def frame_idx(p: Path) -> int:
    return int(re.search(r"(\d+)$", p.stem).group(1))


def unflip(kp: np.ndarray, width: int) -> np.ndarray:
    out = kp[FLIP_IDX].copy()
    out[:, 0] = width - out[:, 0]
    return out


# --------------------------------------------------------------------------------------------- inference
def run(root: Path, onnx: str, yolo_w: str, out: Path, limit: int | None) -> None:
    import cv2
    from ultralytics import YOLO

    from pose_common import OnnxVitPose, iou

    vit = OnnxVitPose(onnx)
    yolo = YOLO(yolo_w)
    done = set()
    if out.is_file():
        done = {json.loads(ln)["file"] for ln in out.read_text().splitlines() if ln.strip()}
    imgs = sorted([p for sp in ("train", "val") for p in (root / "images" / sp).glob("*.jpg")], key=frame_idx)
    if limit:
        imgs = imgs[:limit]
    todo = [p for p in imgs if p.name not in done]
    print(f"{len(imgs)} frames, {len(done)} already done, {len(todo)} to run")

    def yolo_match(res, box):
        if res.keypoints is None or len(res.boxes) == 0:
            return None
        ious = [iou(tuple(b), box) for b in res.boxes.xyxy.cpu().numpy()]
        i = int(np.argmax(ious))
        if ious[i] < 0.3:
            return None
        return np.concatenate([res.keypoints.xy[i].cpu().numpy(), res.keypoints.conf[i].cpu().numpy()[:, None]], 1)

    t0 = time.time()
    with open(out, "a", encoding="utf-8") as f:
        for n, p in enumerate(todo, 1):
            img = cv2.imread(str(p))
            h, w = img.shape[:2]
            t = (p.parent.parent.parent / "labels" / p.parent.name / (p.stem + ".txt")).read_text().split()
            cx, cy, bw, bh = (float(v) for v in t[1:5])
            box = ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)
            fimg = np.ascontiguousarray(img[:, ::-1])
            fbox = (w - box[2], box[1], w - box[0], box[3])
            kv = vit.predict(img, [box])[0]
            kvf = unflip(vit.predict(fimg, [fbox])[0], w)
            r = yolo.predict([img, fimg], verbose=False)
            ky = yolo_match(r[0], box)
            kyf = yolo_match(r[1], fbox)
            kyf = unflip(kyf, w) if kyf is not None else None
            idx = frame_idx(p)
            rec = {"file": p.name, "idx": idx, "folder": p.parent.name, "block": idx // BLOCK,
                   "heldout": idx // BLOCK in HELDOUT_BLOCKS, "box": [round(v, 1) for v in box], "W": w, "H": h,
                   "vit": np.round(kv, 3).tolist(), "vit_f": np.round(kvf, 3).tolist(),
                   "yolo": None if ky is None else np.round(ky, 3).tolist(),
                   "yolo_f": None if kyf is None else np.round(kyf, 3).tolist()}
            f.write(json.dumps(rec) + "\n")
            if n % 250 == 0 or n == len(todo):
                f.flush()
                el = time.time() - t0
                print(f"  {n}/{len(todo)}  {el / n * 1000:.0f} ms/frame  eta {el / n * (len(todo) - n) / 60:.1f} min")


# --------------------------------------------------------------------------------------------- analysis
def analyze(pred: Path, out_dir: Path, conf: float, root: Path | None) -> dict:
    recs = sorted((json.loads(ln) for ln in pred.read_text().splitlines() if ln.strip()), key=lambda r: r["idx"])
    for r in recs:
        for m in ("vit", "vit_f", "yolo", "yolo_f"):
            r[m] = None if r[m] is None else np.array(r[m], dtype=float)
        x1, y1, x2, y2 = r["box"]
        r["size"] = max(x2 - x1, y2 - y1, 1.0)
        r["c"] = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    K = 17

    def ok(k, i):
        return k is not None and k[i][2] >= conf

    def d(a, b, i, s):
        return float(np.hypot(*(a[i][:2] - b[i][:2]))) / s

    cov = {m: np.zeros(K) for m in ("vit", "yolo")}
    flip = {m: [[] for _ in range(K)] for m in ("vit", "yolo")}
    cross = [[] for _ in range(K)]
    for r in recs:
        for i in range(K):
            for m in ("vit", "yolo"):
                if ok(r[m], i):
                    cov[m][i] += 1
                    if ok(r[m + "_f"], i):
                        flip[m][i].append(d(r[m], r[m + "_f"], i, r["size"]))
            if ok(r["vit"], i) and ok(r["yolo"], i):
                cross[i].append(d(r["vit"], r["yolo"], i, r["size"]))
    n = len(recs)

    # runs of near-consecutive frames (gap <= 3) -> jitter = second difference after removing box-centre motion
    runs, cur = [], [recs[0]]
    for a, b in zip(recs, recs[1:]):
        if b["idx"] - a["idx"] <= 3:
            cur.append(b)
        else:
            runs.append(cur)
            cur = [b]
    runs.append(cur)
    runs = [r for r in runs if len(r) >= 3]
    jit = {m: [[] for _ in range(K)] for m in ("vit", "yolo")}
    for run_ in runs:
        for a, b, c in zip(run_, run_[1:], run_[2:]):
            for m in ("vit", "yolo"):
                for i in range(K):
                    if ok(a[m], i) and ok(b[m], i) and ok(c[m], i):
                        pa, pb, pc = a[m][i][:2] - a["c"], b[m][i][:2] - b["c"], c[m][i][:2] - c["c"]
                        jit[m][i].append(float(np.hypot(*(pa - 2 * pb + pc))) / b["size"])

    # confidence vs agreement: does a confident point agree with the other model?
    calib = {}
    for m, o in (("vit", "yolo"), ("yolo", "vit")):
        bins = {"0.5-0.7": [], "0.7-0.9": [], ">=0.9": []}
        for r in recs:
            if r[m] is None or r[o] is None:
                continue
            for i in range(K):
                c_ = r[m][i][2]
                if c_ < conf or r[o][i][2] < 0.3:
                    continue
                key = "0.5-0.7" if c_ < 0.7 else ("0.7-0.9" if c_ < 0.9 else ">=0.9")
                bins[key].append(d(r[m], r[o], i, r["size"]) <= CROSS_OK)
        calib[m] = {k: (round(float(np.mean(v)), 3), len(v)) for k, v in bins.items() if v}

    # consensus (E1 candidate labels) and disputes
    prev = {}
    by_idx = {r["idx"]: r for r in recs}
    for r in recs:
        p = next((by_idx[r["idx"] - g] for g in (1, 2, 3) if r["idx"] - g in by_idx), None)
        prev[r["idx"]] = p
    cons_cov = np.zeros(K)
    n_train = 0
    cons_per_frame, disputes = [], []
    for r in recs:
        npass, ndis = 0, 0
        for i in range(K):
            if ok(r["vit"], i) and ok(r["yolo"], i):
                dc = d(r["vit"], r["yolo"], i, r["size"])
                if dc > DISPUTE:
                    ndis += 1
                good = dc <= CROSS_OK
                for m in ("vit", "yolo"):
                    good = good and ok(r[m + "_f"], i) and d(r[m], r[m + "_f"], i, r["size"]) <= FLIP_OK
                p = prev[r["idx"]]
                if good and p is not None and ok(p["vit"], i):
                    jump = float(np.hypot(*((r["vit"][i][:2] - r["c"]) - (p["vit"][i][:2] - p["c"])))) / r["size"]
                    good = jump <= JUMP_OK
                if good:
                    npass += 1
                    if not r["heldout"]:
                        cons_cov[i] += 1
        if not r["heldout"]:
            n_train += 1
            cons_per_frame.append(npass)
        disputes.append((ndis, r))
    cons_cov /= max(n_train, 1)
    cpf = np.array(cons_per_frame) if cons_per_frame else np.zeros(1)
    lower_cov = float(cons_cov[LOWER_BODY].mean())
    feasible = float(np.median(cpf)) >= 8 and lower_cov >= 0.30

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = ["keypoint,cov_vit,cov_yolo,flip_med_vit,flip_med_yolo,cross_med,cross_le_0.10,"
            "jitter_med_vit,jitter_med_yolo,consensus_cov_train"]

    def med(x):
        return f"{np.median(x):.3f}" if x else ""

    for i, nm in enumerate(COCO17):
        cr = cross[i]
        rows.append(",".join([nm, f"{cov['vit'][i] / n:.3f}", f"{cov['yolo'][i] / n:.3f}", med(flip["vit"][i]),
                              med(flip["yolo"][i]), med(cr),
                              f"{np.mean(np.array(cr) <= CROSS_OK):.3f}" if cr else "", med(jit["vit"][i]),
                              med(jit["yolo"][i]), f"{cons_cov[i]:.3f}"]))
    (out_dir / "e0_per_keypoint.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    held = sorted([x for x in disputes if x[1]["heldout"]], key=lambda x: -x[0])[:30]
    (out_dir / "e0_disputed_heldout.csv").write_text(
        "file,idx,block,disputed_keypoints\n" + "".join(f"{r['file']},{r['idx']},{r['block']},{k}\n" for k, r in held),
        encoding="utf-8")

    no_yolo = sum(r["yolo"] is None for r in recs)
    rep = [
        "E0 LABEL-FREE DIAGNOSTICS - consistency/agreement only, NOT accuracy",
        f"frames {n} (held-out {n - n_train}, training-eligible {n_train}); runs >= 3 frames: {len(runs)}; "
        f"YOLO found no matching person in {no_yolo} frames",
        f"thresholds: conf >= {conf}; agree <= {CROSS_OK}; flip <= {FLIP_OK}; jump <= {JUMP_OK}; "
        f"dispute > {DISPUTE} (fractions of box size)",
        "",
        "per keypoint: see e0_per_keypoint.csv (coverage, flip error, cross-model distance, jitter, consensus)",
        "",
        "confidence vs agreement with the other model (fraction within the agree threshold, count):",
        f"  ViTPose: {calib.get('vit')}",
        f"  YOLO:    {calib.get('yolo')}",
        "",
        f"consensus keypoints per training-eligible frame: median {np.median(cpf):.1f}, mean {cpf.mean():.1f} of 17",
        f"consensus coverage upper body {cons_cov[UPPER_BODY].mean():.2f}, lower body {lower_cov:.2f}",
        f"E1 feasibility (proposed rule: median >= 8 AND lower body >= 0.30): {'FEASIBLE' if feasible else 'NOT FEASIBLE'}",
        "",
        f"most disputed held-out frames (E3 candidates): e0_disputed_heldout.csv (top {len(held)})",
    ]
    (out_dir / "e0_report.txt").write_text("\n".join(rep) + "\n", encoding="utf-8")
    print("\n".join(rep))
    print()
    print("\n".join(rows))

    if root is not None and held:
        sheet(root, held[:12], out_dir / "e0_disputed_heldout.jpg", conf)
    return {"feasible": feasible, "median_consensus": float(np.median(cpf)), "lower_cov": lower_cov}


def sheet(root: Path, items, path: Path, conf: float) -> None:
    import cv2

    sk = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]

    def draw(im, k, col):
        if k is None:
            return
        for a, b in sk:
            if k[a][2] >= conf and k[b][2] >= conf:
                cv2.line(im, tuple(int(v) for v in k[a][:2]), tuple(int(v) for v in k[b][:2]), col, 3)
        for x, y, c in k:
            if c >= conf:
                cv2.circle(im, (int(x), int(y)), 6, col, -1)

    tiles = []
    for _, r in items:
        p = next((root / "images" / sp / r["file"] for sp in ("train", "val")
                  if (root / "images" / sp / r["file"]).is_file()), None)
        if p is None:
            continue
        im = cv2.imread(str(p))
        draw(im, r["vit"], (0, 255, 255))   # ViTPose yellow
        draw(im, r["yolo"], (255, 0, 255))  # YOLO magenta
        cx, cy = (int(v) for v in r["c"])
        s = int(r["size"] * 0.8)
        crop = cv2.resize(im[max(0, cy - s):cy + s, max(0, cx - s):cx + s], (300, 300))
        cv2.putText(crop, r["file"][6:11], (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        tiles.append(crop)
    while len(tiles) % 6:
        tiles.append(np.zeros((300, 300, 3), np.uint8))
    rows = [np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]
    cv2.imwrite(str(path), np.vstack(rows))
    print("saved", path, "(yellow = ViTPose, magenta = YOLO11s-pose)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--root", type=Path, help="dataset root with images/{train,val} and labels/{train,val}")
    ap.add_argument("--onnx")
    ap.add_argument("--yolo")
    ap.add_argument("--pred", type=Path, required=True, help="predictions JSONL (written by --run, read by --analyze)")
    ap.add_argument("--out", type=Path, help="analysis output folder")
    ap.add_argument("--limit", type=int, help="only the first N frames (smoke test)")
    ap.add_argument("--conf", type=float, default=0.5)
    a = ap.parse_args()
    if a.run:
        run(a.root, a.onnx, a.yolo, a.pred, a.limit)
    if a.analyze:
        analyze(a.pred, a.out or a.pred.parent, a.conf, a.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
