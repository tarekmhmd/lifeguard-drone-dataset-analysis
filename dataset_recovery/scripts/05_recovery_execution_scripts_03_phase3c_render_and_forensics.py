"""Phase 3C: Deterministic pixel/box-seam forensics and contact-sheet / overlay generation
across all 4 batches of the 737 KEEP_IMAGE_REVIEW_LABEL images:
  - Batch 3C-1: SF02_output_NNNN (39 images) -> 100% rendered on 3 contact sheets
  - Batch 3C-2: SF12_clipchamp_mp4 (40 images) -> 100% rendered on 3 contact sheets
  - Batch 3C-3: SF18_other_static (86 images) -> 100% rendered on 6 contact sheets
  - Batch 3C-4: SF04_pure_digits (572 images) -> all un-rotated/low-seam + stratified across all prefixes on 6 contact sheets
Strict read-only access to original ZIP archives.
"""
import hashlib, os, re, zipfile
import cv2, numpy as np, pandas as pd

P2_DIR = r"E:\lifeguardDron\DrowningDetection_v1_cleaning\04_recovery_investigation"
P3_DIR = r"E:\lifeguardDron\DrowningDetection_v1_cleaning\05_recovery_execution"
CS_DIR = os.path.join(P3_DIR, "03_label_review_737", "contact_sheets")
OV_DIR = os.path.join(P3_DIR, "03_label_review_737", "overlays")
os.makedirs(CS_DIR, exist_ok=True)
os.makedirs(OV_DIR, exist_ok=True)

SRC_DIR = r"E:\lifeguardDron\DrowningDetectionTracking_v1"
ZIPS = {
    "train": zipfile.ZipFile(os.path.join(SRC_DIR, "train-20261102T163226Z-1-001.zip"), "r"),
    "valid": zipfile.ZipFile(os.path.join(SRC_DIR, "valid-20261102T163228Z-1-001.zip"), "r"),
    "test": zipfile.ZipFile(os.path.join(SRC_DIR, "test-20261102T163222Z-1-001.zip"), "r"),
}

inv = pd.read_csv(os.path.join(P2_DIR, "FULL_DATASET_INVENTORY.csv"), low_memory=False)
rec = pd.read_csv(os.path.join(P2_DIR, "RECOVERY_CANDIDATES.csv"), low_memory=False)
inv_keys = pd.read_csv(os.path.join(P2_DIR, "analysis_logs", "cache", "inventory_keys.csv"), low_memory=False)
boxes_raw = pd.read_csv(os.path.join(P2_DIR, "analysis_logs", "cache", "boxes_raw.csv"), low_memory=False)

df737 = rec[rec["recovery_decision"] == "KEEP_IMAGE_REVIEW_LABEL"].merge(
    inv[["rel_image_path", "filename", "image_sha256", "annotation_sha256", "label_path", "file_size", "sha_group_size", "annotation_line_count"]],
    on="rel_image_path", how="left"
).merge(
    inv_keys[["rel_image_path", "image_id", "aqua_frac", "blue_frac", "green_frac", "dark_frac", "sp_frac_global", "seam_row", "seam_col", "black_corners", "phash"]],
    on="rel_image_path", how="left"
).sort_values(["super_family", "source_prefix", "rel_image_path"]).reset_index(drop=True)

print("Loaded 737 KEEP_IMAGE_REVIEW_LABEL images:", len(df737), df737["super_family"].value_counts().to_dict())

def compute_box_seam_and_rot(gray, ib):
    """Measure (1) border black wedge fraction (global rotation/letterbox) and
    (2) bounding-box perimeter discontinuity ratio (`box_seam_ratio`), which detects
    baked-in Roboflow bounding-box-level rotation/shear patches around ground-truth boxes.
    """
    H, W = gray.shape
    g = gray.astype(np.float32)
    # Border black fraction along outer 8-pixel frame
    bmask = np.zeros((H, W), dtype=bool)
    bmask[:8, :] = True; bmask[-8:, :] = True; bmask[:, :8] = True; bmask[:, -8:] = True
    border_blk_frac = float((gray[bmask] < 12).mean())

    seam_scores = []
    for _, bx in ib.iterrows():
        if pd.isna(bx["cx"]):
            continue
        cx, cy, bw, bh = float(bx["cx"]), float(bx["cy"]), float(bx["w"]), float(bx["h"])
        x1 = int(round((cx - bw / 2) * W))
        y1 = int(round((cy - bh / 2) * H))
        x2 = int(round((cx + bw / 2) * W))
        y2 = int(round((cy + bh / 2) * H))
        if x2 - x1 < 16 or y2 - y1 < 16:
            continue
        edges = []
        # Top edge y1
        if 6 <= y1 < H - 6:
            xa, xb = max(4, x1 + 4), min(W - 4, x2 - 4)
            if xb > xa:
                strip = np.abs(np.diff(g[y1 - 5 : y1 + 5, xa:xb], axis=0)).mean(axis=1) # len 9, index 4 is y1-1..y1
                peak = float(strip[3:6].max())
                bg = float(np.median(np.concatenate([strip[:3], strip[6:]]))) + 1.0
                edges.append(peak / bg)
        # Bottom edge y2
        if 6 <= y2 < H - 6:
            xa, xb = max(4, x1 + 4), min(W - 4, x2 - 4)
            if xb > xa:
                strip = np.abs(np.diff(g[y2 - 5 : y2 + 5, xa:xb], axis=0)).mean(axis=1)
                peak = float(strip[3:6].max())
                bg = float(np.median(np.concatenate([strip[:3], strip[6:]]))) + 1.0
                edges.append(peak / bg)
        # Left edge x1
        if 6 <= x1 < W - 6:
            ya, yb = max(4, y1 + 4), min(H - 4, y2 - 4)
            if yb > ya:
                strip = np.abs(np.diff(g[ya:yb, x1 - 5 : x1 + 5], axis=1)).mean(axis=0)
                peak = float(strip[3:6].max())
                bg = float(np.median(np.concatenate([strip[:3], strip[6:]]))) + 1.0
                edges.append(peak / bg)
        # Right edge x2
        if 6 <= x2 < W - 6:
            ya, yb = max(4, y1 + 4), min(H - 4, y2 - 4)
            if yb > ya:
                strip = np.abs(np.diff(g[ya:yb, x2 - 5 : x2 + 5], axis=1)).mean(axis=0)
                peak = float(strip[3:6].max())
                bg = float(np.median(np.concatenate([strip[:3], strip[6:]]))) + 1.0
                edges.append(peak / bg)
        if edges:
            seam_scores.append(float(np.mean(edges)))
    max_seam = round(max(seam_scores), 3) if seam_scores else 0.0
    mean_seam = round(float(np.mean(seam_scores)), 3) if seam_scores else 0.0
    return round(border_blk_frac, 4), max_seam, mean_seam

COLORS = {5: (0, 0, 255), 6: (255, 128, 0), 0: (0, 165, 255), 1: (255, 0, 255), 2: (255, 255, 0), 3: (0, 255, 255), 4: (0, 255, 0)}
CNAMES = {5: "5:drown", 6: "6:swim", 0: "0:'3'", 1: "1:'4'", 2: "2:'5'", 3: "3:'6'", 4: "4:OutW"}

from collections import defaultdict
records = []
tiles_by_sf = defaultdict(list)

for idx, r in df737.iterrows():
    cid_str = f"R737_{idx+1:03d}"
    sp = r["original_split"]
    rel = r["rel_image_path"]
    lpath = r["label_path"]
    sf = r["super_family"]
    raw_bytes = ZIPS[sp].read(rel)
    lbl_bytes = ZIPS[sp].read(lpath)
    sha_ok = int(hashlib.sha256(raw_bytes).hexdigest() == r["image_sha256"])
    lsha_ok = int(hashlib.sha256(lbl_bytes).hexdigest() == r["annotation_sha256"])
    img = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = img.shape[:2]
    ib = boxes_raw[(boxes_raw["split"] == sp) & (boxes_raw["rel_image_path"] == rel)]
    b_blk, max_seam, mean_seam = compute_box_seam_and_rot(gray, ib)

    # Render annotated overlay tile
    over = img.copy()
    b_list = []
    for _, bx in ib.iterrows():
        cid = int(bx["class_id"])
        cx, cy, bw, bh = float(bx["cx"]), float(bx["cy"]), float(bx["w"]), float(bx["h"])
        x1, y1 = int(round((cx - bw / 2) * W)), int(round((cy - bh / 2) * H))
        x2, y2 = int(round((cx + bw / 2) * W)), int(round((cy + bh / 2) * H))
        col = COLORS.get(cid, (0, 255, 255))
        cv2.rectangle(over, (x1, y1), (x2, y2), col, 2)
        cv2.putText(over, f"#{int(bx['line_no'])} {CNAMES.get(cid, str(cid))}",
                    (max(2, x1), max(16, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 2, cv2.LINE_AA)
        b_list.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    # Build 360x360 contact sheet tile
    t_img = cv2.resize(over, (360, 360), interpolation=cv2.INTER_AREA)
    t_ban = np.zeros((44, 360, 3), dtype=np.uint8)
    cv2.putText(t_ban, f"{cid_str} [{sp}] {r['source_prefix'][:26]}",
                (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(t_ban, f"cls={r['class_ids']} ({len(ib)}b) crn={r['black_corners']} bBlk={b_blk:.2f} seam={max_seam:.2f}",
                (4, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (180, 235, 255), 1, cv2.LINE_AA)
    tile = np.vstack([t_ban, t_img])

    rec_item = {
        "review_id": cid_str,
        "image_id": int(r["image_id"]),
        "original_split": sp,
        "rel_image_path": rel,
        "label_path": lpath,
        "source_prefix": r["source_prefix"],
        "source_group": r["source_group"],
        "super_family": sf,
        "sha_group_id": int(r["sha_group_id"]),
        "sha_group_size": int(r["sha_group_size"]),
        "image_sha256": r["image_sha256"],
        "sha256_verified": sha_ok,
        "annotation_sha256": r["annotation_sha256"],
        "label_sha256_verified": lsha_ok,
        "n_boxes": len(ib),
        "class_ids": str(r["class_ids"]),
        "max_box_area": float(ib["area"].max()) if len(ib) else 0.0,
        "min_box_area": float(ib["area"].min()) if len(ib) else 0.0,
        "max_in_dark": float(ib["in_dark"].max()) if len(ib) else 0.0,
        "max_in_sp": float(ib["in_sp"].max()) if len(ib) else 0.0,
        "black_corners": int(r["black_corners"]),
        "border_black_frac": b_blk,
        "max_box_seam_ratio": max_seam,
        "mean_box_seam_ratio": mean_seam,
        "aqua_frac": float(r["aqua_frac"]),
        "blue_frac": float(r["blue_frac"]),
        "original_boxes_yolo": " ; ".join(b_list),
    }
    records.append(rec_item)
    tiles_by_sf[sf].append((rec_item, tile, img, over))

def write_sheets(prefix_name, tile_tuples, per_sheet=16):
    sheet_map = {}
    for s_idx in range(0, len(tile_tuples), per_sheet):
        chunk = tile_tuples[s_idx : s_idx + per_sheet]
        s_num = (s_idx // per_sheet) + 1
        sname = f"{prefix_name}_sheet_{s_num:02d}.jpg"
        t_list = [c[1] for c in chunk]
        for t_i, c in enumerate(chunk, 1):
            sheet_map[c[0]["review_id"]] = (sname, t_i)
        while len(t_list) < per_sheet:
            t_list.append(np.zeros_like(t_list[0]))
        grid = np.vstack([np.hstack(t_list[r * 4 : (r + 1) * 4]) for r in range(4)])
        cv2.imwrite(os.path.join(CS_DIR, sname), grid, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        print("Wrote", sname, len(chunk))
    return sheet_map

smap = {}
# Batch 3C-1: 100% of SF02_output_NNNN in 737 (51 images -> 4 sheets)
smap.update(write_sheets("b3c1_sf02_output", tiles_by_sf["SF02_output_NNNN"], 16))
# Batch 3C-2: 100% of SF12_clipchamp_mp4 (41 images) + SF08_drowningN_mp4 (1 image) = 42 images -> 3 sheets
b3c2_tiles = tiles_by_sf["SF12_clipchamp_mp4"] + tiles_by_sf["SF08_drowningN_mp4"]
smap.update(write_sheets("b3c2_sf12_clipchamp", b3c2_tiles, 16))

# Separate SF04_pure_digits (644 images) into:
#   - Batch 3C-3: Non-1..200_jpg sub-series in SF04 (e.g. 10000..12000_jpg, 130000_jpg, etc.) + un-rotated/low-seam frames
#   - Batch 3C-4: 1..200_jpg series (stratified across all prefixes)
sf04_all = tiles_by_sf["SF04_pure_digits"]
def prefix_num(c):
    m = re.match(r"^(\d+)_", c[0]["source_prefix"])
    return int(m.group(1)) if m else 999999

sf04_high_num = [c for c in sf04_all if prefix_num(c) > 250]
sf04_low_num = [c for c in sf04_all if prefix_num(c) <= 250]
sf04_low_unrot = [c for c in sf04_low_num if c[0]["black_corners"] == 0 and c[0]["border_black_frac"] < 0.02]
sf04_low_rot = [c for c in sf04_low_num if not (c[0]["black_corners"] == 0 and c[0]["border_black_frac"] < 0.02)]
print(f"SF04 breakdown: high_num (>250)={len(sf04_high_num)}, low_num (<=250)={len(sf04_low_num)} (unrot={len(sf04_low_unrot)}, rot={len(sf04_low_rot)})")

# Batch 3C-3: All sf04_high_num (e.g. 10000..130500_jpg) + all sf04_low_unrot (up to 96 tiles = 6 sheets)
b3c3_candidates = sf04_high_num + sf04_low_unrot
if len(b3c3_candidates) < 80:
    # Fill remaining slots of Batch 3C-3 with lowest seam-score frames from sf04_low_rot
    rem_rot = sorted(sf04_low_rot, key=lambda c: (c[0]["max_box_seam_ratio"], c[0]["black_corners"]))
    b3c3_pick = b3c3_candidates + rem_rot[: (80 - len(b3c3_candidates))]
else:
    b3c3_pick = b3c3_candidates[:96]
smap.update(write_sheets("b3c3_sf04_high_and_unrot", b3c3_pick, 16))

# Batch 3C-4: Stratified across the remaining sf04_low_num (1..200_jpg) across 6 sheets (96 tiles)
used_ids = {c[0]["review_id"] for c in b3c3_pick}
rem_low = [c for c in sf04_low_num if c[0]["review_id"] not in used_ids]
step4 = max(1, len(rem_low) // 96)
b3c4_pick = [rem_low[i * step4] for i in range(min(96, len(rem_low)))]
smap.update(write_sheets("b3c4_sf04_digits_1to200", b3c4_pick, 16))

# Also save 24 full-resolution side-by-side (raw | overlay) diagnostic images across all batches in overlays/
diag_picks = (
    tiles_by_sf["SF02_output_NNNN"][:6]
    + b3c2_tiles[:6]
    + b3c3_pick[:6]
    + b3c4_pick[:6]
)
for rec_item, _, raw_im, ov_im in diag_picks:
    rid = rec_item["review_id"]
    ban = np.zeros((50, 1280, 3), dtype=np.uint8)
    cv2.putText(ban, f"{rid} | [{rec_item['original_split']}] {rec_item['rel_image_path']} | crn={rec_item['black_corners']} bBlk={rec_item['border_black_frac']:.3f} seam={rec_item['max_box_seam_ratio']:.2f}",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(ban, f"sf={rec_item['super_family']} | cls={rec_item['class_ids']} | boxes={rec_item['n_boxes']} | {rec_item['original_boxes_yolo'][:95]}",
                (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 235, 255), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(OV_DIR, f"{rid}_{rec_item['source_prefix'][:30]}.jpg"), np.vstack([ban, np.hstack([raw_im, ov_im])]), [int(cv2.IMWRITE_JPEG_QUALITY), 92])

for r in records:
    sh, ti = smap.get(r["review_id"], ("", 0))
    r["contact_sheet"] = sh
    r["tile_index"] = ti

rec_df = pd.DataFrame(records)
rec_df.to_csv(os.path.join(P3_DIR, "logs", "phase3c_737_pre_review.csv"), index=False, encoding="utf-8")
for zf in ZIPS.values():
    zf.close()
print("Saved logs/phase3c_737_pre_review.csv:", len(rec_df), "Visual tiles rendered:", len(smap))
