"""Phase B/D/E core extraction: one read-only pass over the three original ZIP archives.

Outputs (all inside 04_recovery_investigation/analysis_logs/cache/):
  inventory_raw.csv   one row per image
  boxes_raw.csv       one row per annotation line (detection or polygon or bad)
  features.npz        per-image descriptors (phash, dhash, flipped phash, color layout, hsv hist)
Original archives are opened in read mode only. Nothing is written next to the source data.
"""
import csv, hashlib, io, os, re, sys, zipfile, time
import numpy as np
import cv2
from concurrent.futures import ProcessPoolExecutor

SRC = r"E:\lifeguardDron\DrowningDetectionTracking_v1"
OUT = r"E:\lifeguardDron\DrowningDetection_v1_cleaning\04_recovery_investigation\analysis_logs\cache"
ZIPS = {
    "train": os.path.join(SRC, "train-20261102T163226Z-1-001.zip"),
    "valid": os.path.join(SRC, "valid-20261102T163228Z-1-001.zip"),
    "test": os.path.join(SRC, "test-20261102T163222Z-1-001.zip"),
}
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

_z = {}

def _init():
    for k, v in ZIPS.items():
        _z[k] = zipfile.ZipFile(v, "r")

def phash64(gray):
    g = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(g)[:8, :8].flatten()
    med = np.median(d[1:])
    bits = (d > med).astype(np.uint8)
    return int("".join(map(str, bits)), 2)

def dhash64(gray):
    g = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten().astype(np.uint8)
    return int("".join(map(str, bits)), 2)

def seam_score(gray):
    """Mosaic cue: strongest full-span row/column discontinuity that appears in BOTH halves."""
    g = cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)
    H, W = g.shape
    dr = np.abs(np.diff(g, axis=0))  # (H-1, W)
    dc = np.abs(np.diff(g, axis=1))  # (H, W-1)
    m = int(0.06 * H)
    rl = dr[:, : W // 2].mean(1); rr = dr[:, W // 2 :].mean(1)
    cl = dc[: H // 2, :].mean(0); cb = dc[H // 2 :, :].mean(0)
    rows = np.minimum(rl, rr); cols = np.minimum(cl, cb)
    def best(v):
        inner = v[m:-m]
        base = np.median(v) + 1e-3
        i = int(np.argmax(inner))
        return float(inner[i] / base), i + m
    rs, ry = best(rows); cs, cx = best(cols)
    return rs, ry, cs, cx

def box_stats(gray, sp_map, x1, y1, x2, y2):
    H, W = gray.shape
    xa, ya, xb, yb = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if xb - xa < 2 or yb - ya < 2:
        return (np.nan,) * 5
    roi = gray[ya:yb, xa:xb]
    sp = sp_map[ya:yb, xa:xb].mean()
    dark = (roi < 25).mean()
    std = float(roi.std())
    # ring around the box (10% margin)
    mx = max(4, int(0.1 * (xb - xa))); my = max(4, int(0.1 * (yb - ya)))
    ra, rb, rc, rd = max(0, xa - mx), max(0, ya - my), min(W, xb + mx), min(H, yb + my)
    ring_mask = np.ones((rd - rb, rc - ra), bool)
    ring_mask[ya - rb : yb - rb, xa - ra : xb - ra] = False
    ring_sp = sp_map[rb:rd, ra:rc][ring_mask].mean() if ring_mask.any() else np.nan
    return float(sp), float(dark), std, float(ring_sp), float(roi.mean())

def process(item):
    split, name, lname = item
    zf = _z[split]
    rec = {"split": split, "rel_image_path": name, "filename": os.path.basename(name)}
    data = zf.read(name)
    rec["file_size"] = len(data)
    rec["image_sha256"] = hashlib.sha256(data).hexdigest()
    boxes = []
    feat = None
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        rec.update(decode_ok=0, width="", height="")
    else:
        H, W = arr.shape[:2]
        rec.update(decode_ok=1, width=W, height=H)
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        rec["phash"] = format(phash64(gray), "016x")
        rec["phash_flip"] = format(phash64(cv2.flip(gray, 1)), "016x")
        rec["dhash"] = format(dhash64(gray), "016x")
        hsv = cv2.cvtColor(arr, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        rec["aqua_frac"] = round(float(((h >= 80) & (h <= 105) & (s >= 60) & (v >= 90)).mean()), 4)
        rec["blue_frac"] = round(float(((h >= 95) & (h <= 125) & (s >= 60) & (v >= 60)).mean()), 4)
        rec["green_frac"] = round(float(((h >= 35) & (h < 80) & (s >= 50) & (v >= 50)).mean()), 4)
        rec["dark_frac"] = round(float((gray < 15).mean()), 4)
        rec["mean_sat"] = round(float(s.mean()), 2)
        rec["mean_val"] = round(float(v.mean()), 2)
        med = cv2.medianBlur(gray, 3)
        sp_map = (np.abs(gray.astype(np.int16) - med.astype(np.int16)) > 60).astype(np.float32)
        rec["sp_frac_global"] = round(float(sp_map.mean()), 4)
        rs, ry, cs, cx = seam_score(gray)
        rec.update(seam_row=round(rs, 2), seam_row_y=ry, seam_col=round(cs, 2), seam_col_x=cx)
        # corner black triangles (rotation) : mean of the 4 corner 24x24 patches
        c = 24
        corners = [gray[:c, :c], gray[:c, -c:], gray[-c:, :c], gray[-c:, -c:]]
        rec["black_corners"] = int(sum(int(p.mean() < 12) for p in corners))
        lab = cv2.cvtColor(cv2.resize(arr, (8, 8), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2LAB)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 4, 4], [0, 180, 0, 256, 0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-9)
        small = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA).flatten()
        feat = (lab.flatten().astype(np.uint8), hist.astype(np.float32), small.astype(np.uint8))
    # label
    rec["label_path"] = lname or ""
    if lname is None:
        rec.update(label_exists=0, label_sha256="", label_line_count=0, label_format="MISSING",
                   n_det=0, n_poly=0, n_bad=0, class_ids="")
    else:
        lb = zf.read(lname)
        rec["label_exists"] = 1
        rec["label_sha256"] = hashlib.sha256(lb).hexdigest()
        lines = [ln for ln in lb.decode("utf-8", "replace").splitlines() if ln.strip()]
        rec["label_line_count"] = len(lines)
        nd = npl = nb = 0
        cls = set()
        for i, ln in enumerate(lines, 1):
            t = ln.split()
            b = {"split": split, "rel_image_path": name, "line_no": i, "n_tokens": len(t)}
            try:
                vals = [float(x) for x in t]
                c0 = vals[0]
                cid_ok = c0.is_integer() and 0 <= c0 <= 6
            except ValueError:
                vals = None; cid_ok = False
            if vals is None:
                b["line_format"] = "UNPARSEABLE"; nb += 1
            elif len(t) == 5:
                b["line_format"] = "DETECTION"; nd += 1
            elif len(t) >= 7 and len(t) % 2 == 1:
                b["line_format"] = "POLYGON"; npl += 1
            else:
                b["line_format"] = "UNKNOWN_TOKENS"; nb += 1
            b["class_id"] = int(vals[0]) if (vals and cid_ok) else (t[0] if t else "")
            b["class_valid"] = int(bool(cid_ok))
            if vals and cid_ok:
                cls.add(int(vals[0]))
            if vals and b["line_format"] in ("DETECTION", "POLYGON"):
                if b["line_format"] == "DETECTION":
                    cx_, cy_, w_, h_ = vals[1:5]
                    x1n, y1n, x2n, y2n = cx_ - w_ / 2, cy_ - h_ / 2, cx_ + w_ / 2, cy_ + h_ / 2
                    coords = vals[1:5]
                else:
                    xs = vals[1::2]; ys = vals[2::2]
                    x1n, y1n, x2n, y2n = min(xs), min(ys), max(xs), max(ys)
                    cx_, cy_, w_, h_ = (x1n + x2n) / 2, (y1n + y2n) / 2, x2n - x1n, y2n - y1n
                    coords = vals[1:]
                    b["n_vertices"] = len(xs)
                b.update(cx=round(cx_, 6), cy=round(cy_, 6), w=round(w_, 6), h=round(h_, 6),
                         area=round(max(w_, 0) * max(h_, 0), 6))
                b["coords_in_unit"] = int(all(0 <= v <= 1 for v in coords))
                eps = 1e-6
                b["oob"] = int(x1n < -eps or y1n < -eps or x2n > 1 + eps or y2n > 1 + eps)
                b["nonpositive_wh"] = int(w_ <= 0 or h_ <= 0)
                b["touch_border"] = int(x1n <= 0.005 or y1n <= 0.005 or x2n >= 0.995 or y2n >= 0.995)
                if arr is not None:
                    Hh, Ww = arr.shape[:2]
                    st = box_stats(gray, sp_map, int(round(x1n * Ww)), int(round(y1n * Hh)),
                                   int(round(x2n * Ww)), int(round(y2n * Hh)))
                    b.update(in_sp=st[0], in_dark=st[1], in_std=st[2], ring_sp=st[3], in_mean=st[4])
            boxes.append(b)
        rec.update(n_det=nd, n_poly=npl, n_bad=nb, class_ids=";".join(map(str, sorted(cls))))
        if not lines:
            rec["label_format"] = "EMPTY"
        elif nb:
            rec["label_format"] = "HAS_UNKNOWN"
        elif npl and nd:
            rec["label_format"] = "MIXED_DET_POLY"
        elif npl:
            rec["label_format"] = "POLYGON_ONLY"
        else:
            rec["label_format"] = "DETECTION"
    return rec, boxes, feat

def main():
    os.makedirs(OUT, exist_ok=True)
    items = []
    stats = {}
    for split, zp in ZIPS.items():
        with zipfile.ZipFile(zp) as zf:
            names = zf.namelist()
        imgs = [n for n in names if n.lower().endswith(IMG_EXT) and "/images/" in n]
        labs = {n for n in names if n.lower().endswith(".txt") and "/labels/" in n}
        lab_by_stem = {os.path.splitext(os.path.basename(n))[0]: n for n in labs}
        img_stems = set()
        for n in sorted(imgs):
            stem = os.path.splitext(os.path.basename(n))[0]
            img_stems.add(stem)
            items.append((split, n, lab_by_stem.get(stem)))
        orphans = [n for s, n in lab_by_stem.items() if s not in img_stems]
        other = [n for n in names if not n.endswith("/") and n not in labs and n not in imgs]
        stats[split] = dict(zip_entries=len(names), images=len(imgs), labels=len(labs),
                            orphan_labels=len(orphans), other_entries=len(other))
        with open(os.path.join(OUT, f"zip_listing_{split}.txt"), "w", encoding="utf-8") as f:
            f.write(f"{stats[split]}\n")
            for o in orphans:
                f.write(f"ORPHAN_LABEL\t{o}\n")
            for o in other:
                f.write(f"OTHER_ENTRY\t{o}\n")
    print(stats, flush=True)
    t0 = time.time()
    recs, allboxes, labf, histf, smallf = [], [], [], [], []
    with ProcessPoolExecutor(max_workers=6, initializer=_init) as ex:
        for i, (rec, boxes, feat) in enumerate(ex.map(process, items, chunksize=64)):
            rec["image_id"] = i
            recs.append(rec); allboxes.extend(boxes)
            if feat is None:
                feat = (np.zeros(192, np.uint8), np.zeros(192, np.float32), np.zeros(256, np.uint8))
            labf.append(feat[0]); histf.append(feat[1]); smallf.append(feat[2])
            if i % 2000 == 0:
                print(i, round(time.time() - t0, 1), flush=True)
    cols = ["image_id", "split", "rel_image_path", "filename", "width", "height", "decode_ok", "file_size",
            "image_sha256", "phash", "phash_flip", "dhash", "label_path", "label_exists", "label_sha256",
            "label_line_count", "label_format", "n_det", "n_poly", "n_bad", "class_ids", "aqua_frac",
            "blue_frac", "green_frac", "dark_frac", "mean_sat", "mean_val", "sp_frac_global", "seam_row",
            "seam_row_y", "seam_col", "seam_col_x", "black_corners"]
    with open(os.path.join(OUT, "inventory_raw.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(recs)
    bcols = ["split", "rel_image_path", "line_no", "n_tokens", "line_format", "class_id", "class_valid",
             "n_vertices", "cx", "cy", "w", "h", "area", "coords_in_unit", "oob", "nonpositive_wh",
             "touch_border", "in_sp", "in_dark", "in_std", "ring_sp", "in_mean"]
    with open(os.path.join(OUT, "boxes_raw.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=bcols, extrasaction="ignore"); w.writeheader(); w.writerows(allboxes)
    np.savez_compressed(os.path.join(OUT, "features.npz"), lab=np.stack(labf), hist=np.stack(histf),
                        small=np.stack(smallf), paths=np.array([r["rel_image_path"] for r in recs]))
    print("DONE", len(recs), len(allboxes), round(time.time() - t0, 1), flush=True)

if __name__ == "__main__":
    main()

