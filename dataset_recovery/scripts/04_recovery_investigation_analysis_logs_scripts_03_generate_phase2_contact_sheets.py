"""Generate 12 structured 4x4 contact sheets (192 new unique images across ~160 distinct video_keys/source_groups)
for Phase C (Scene Classification) and Phase D (Annotation Forensics) visual inspection.
Also generates targeted visual overlays in 04_recovery_investigation/visual_overlays/.
"""
import os, re, zipfile
import cv2, numpy as np, pandas as pd

ROOT = r"E:\lifeguardDron\DrowningDetection_v1_cleaning\04_recovery_investigation"
C = os.path.join(ROOT, "analysis_logs", "cache")
CS_DIR = os.path.join(ROOT, "contact_sheets")
OV_DIR = os.path.join(ROOT, "visual_overlays")
os.makedirs(CS_DIR, exist_ok=True)
os.makedirs(OV_DIR, exist_ok=True)

SRC = r"E:\lifeguardDron\DrowningDetectionTracking_v1"
ZIPS = {
    "train": zipfile.ZipFile(os.path.join(SRC, "train-20261102T163226Z-1-001.zip"), "r"),
    "valid": zipfile.ZipFile(os.path.join(SRC, "valid-20261102T163228Z-1-001.zip"), "r"),
    "test": zipfile.ZipFile(os.path.join(SRC, "test-20261102T163222Z-1-001.zip"), "r"),
}

inv = pd.read_csv(os.path.join(C, "inventory_keys.csv"))
inv["class_ids"] = inv["class_ids"].fillna("")
boxes = pd.read_csv(os.path.join(C, "boxes_raw.csv"), low_memory=False)
prev = pd.read_csv(r"E:\lifeguardDron\DrowningDetection_v1_cleaning\03_source_investigation\VISION_REVIEW_RESULTS.csv")
already = set(prev["image_name"].unique())

# Helper to pick 1 uninspected clean representative image per prefix/video_key
uninspected = inv[~inv["rel_image_path"].isin(already)].copy()

def pick_from_vkeys(vkeys, limit=16, prefer_clean=True):
    rows = []
    for vk in vkeys:
        sub = uninspected[uninspected["video_key"] == vk]
        if sub.empty:
            sub = inv[inv["video_key"] == vk]
        if sub.empty:
            continue
        if prefer_clean:
            sub = sub.sort_values(["sp_frac_global", "dark_frac", "image_id"])
        else:
            sub = sub.sort_values(["image_id"])
        r = sub.iloc[0]
        rows.append(r)
        if len(rows) >= limit:
            break
    return rows

def pick_from_prefixes(prefixes, limit=16, prefer_clean=True):
    rows = []
    for p in prefixes:
        sub = uninspected[uninspected["source_prefix"] == p]
        if sub.empty:
            sub = inv[inv["source_prefix"] == p]
        if sub.empty:
            continue
        if prefer_clean:
            sub = sub.sort_values(["sp_frac_global", "dark_frac", "image_id"])
        else:
            sub = sub.sort_values(["image_id"])
        rows.append(sub.iloc[0])
        if len(rows) >= limit:
            break
    return rows

# Sheet 01: Class 1 non-VisDrone prefixes + KakaoTalk all 8 batches
s01_prefixes = [
    "1_jpg", "5_jpg", "6_jpg", "16_jpg", "19_jpg", "23_jpg",
    "130319_jpg", "130369_jpg", "130394_jpg",
]
s01_vkeys = [
    "kakaotalk_20230504_000949565", "kakaotalk_20230504_162842510",
    "kakaotalk_20230504_000856885", "kakaotalk_20230504_162947731",
    "kakaotalk_20230504_001213619", "kakaotalk_20230509_161408906",
    "kakaotalk_20230504_163038544",
]
sheet01 = pick_from_prefixes(s01_prefixes, 9) + pick_from_vkeys(s01_vkeys, 7)

# Sheet 02: output_NNNN across numeric ranges (output_0002..output_0600 and output1..output50)
out_p = sorted(uninspected[uninspected["video_key"].isin(["output_", "output"])]["source_prefix"].unique())
step = max(1, len(out_p) // 16)
sheet02 = pick_from_prefixes([out_p[i * step] for i in range(min(16, len(out_p)))], 16)

# Sheet 03: drowningN_mp4 (13 videos) + clipchamp (2 videos) + white-water_mp4
drown_vks = sorted(inv[inv["source_prefix"].str.match(r"^drowning\d+_")]["video_key"].unique())
sheet03 = pick_from_vkeys(drown_vks + ["untitled-video-made-with-clipchamp-1-_mp4", "-clipchamp-_mp4", "white-water_mp4"], 16)

# Sheet 04: Same video_key paired comparison: (class 0/3 copy vs class 4/5 copy) across 8 major videos (8 pairs = 16 tiles)
pair_vks = [
    "drowning1_mp4", "drowning47_mp4", "drowning20_mp4", "untitled-video-made-with-clipchamp-1-_mp4",
    "video_2022-12-07_23-08-12_mp4", "3-727-drowning-stock-videos-and-royalty-free-footage-istock_2_mp4",
    "drowning-man-stock-video-download-video-clip-now-drowning-struggle-water-istock_mp4",
    "the-child-is-drowning-in-water-selective-focus-nature-stock-video-download-video-clip-now-istock_mp4",
]
sheet04 = []
for vk in pair_vks:
    sub = inv[inv["video_key"] == vk].sort_values("sp_frac_global")
    r_num = sub[sub["class_ids"].str.contains(r"\b[03]\b", regex=True)]
    r_sem = sub[sub["class_ids"].str.contains(r"\b[45]\b", regex=True)]
    if not r_num.empty and not r_sem.empty:
        sheet04.append(r_num.iloc[0])
        sheet04.append(r_sem.iloc[0])

# Sheet 05: All remaining stock_named videos + video_2022-12-07_* + video40-* + 10_mov
stock_vks = [vk for vk in sorted(inv[inv["source_prefix"].str.contains("istock|shutterstock|bigstock|getty|pond5")]["video_key"].unique()) if vk not in pair_vks]
v22_vks = ["video_2022-12-07_23-08-07_mp4", "video_2022-12-07_23-08-15_mp4", "video_2022-12-07_23-08-17_mp4", "10_mov"]
sheet05 = pick_from_vkeys((stock_vks + v22_vks)[:16], 16)

# Sheet 06: videoN_mp4 (video40-1..6, video6, etc.) + other_video_mp4_mov
vn_vks = sorted(inv[inv["source_prefix"].str.match(r"^video\d+")]["video_key"].unique())
ov_vks = [vk for vk in sorted(inv[inv["video_key_rule"] == "VIDEO_FRAME_SUFFIX"]["video_key"].unique())
          if vk not in set(drown_vks + pair_vks + stock_vks + v22_vks + vn_vks)]
sheet06 = pick_from_vkeys((vn_vks + ov_vks)[:16], 16)

# Sheet 07: imageNNNN family (463 prefixes across range image1..image1999)
img_p = sorted(uninspected[uninspected["source_prefix"].str.match(r"^image\d+_")]["source_prefix"].unique(),
               key=lambda x: int(re.findall(r"\d+", x)[0]))
step7 = max(1, len(img_p) // 16)
sheet07 = pick_from_prefixes([img_p[i * step7] for i in range(min(16, len(img_p)))], 16)

# Sheet 08: pure_digits_jpg_png family across numeric ranges (1..999_png, 10000..12000_jpg, 130000..130400_jpg, 318351193_jpg)
dig_p = sorted(uninspected[uninspected["source_prefix"].str.match(r"^\d+_(jpg|png|jpeg)$")]["source_prefix"].unique(),
               key=lambda x: int(re.findall(r"^\d+", x)[0]))
step8 = max(1, len(dig_p) // 16)
sheet08 = pick_from_prefixes([dig_p[i * step8] for i in range(min(16, len(dig_p)))], 16)

# Sheet 09: val_train_test_N + img_pic_photo families
vt_p = sorted(uninspected[uninspected["source_prefix"].str.match(r"^(val|train|test)\d*_")]["source_prefix"].unique())
ip_p = sorted(uninspected[uninspected["source_prefix"].str.match(r"^(img|images|pic|photo)")]["source_prefix"].unique())
vt_pick = [vt_p[i * max(1, len(vt_p)//8)] for i in range(min(8, len(vt_p)))]
ip_pick = [ip_p[i * max(1, len(ip_p)//8)] for i in range(min(8, len(ip_p)))]
sheet09 = pick_from_prefixes(vt_pick + ip_pick, 16)

# Sheet 10: Top 16 other_static / high-count remaining prefixes (e.g. 134653437-asian-senior..., 53feaf75..., etc.)
used_vks = set(r["video_key"] for s in [sheet01, sheet02, sheet03, sheet04, sheet05, sheet06, sheet07, sheet08, sheet09] for r in s)
rem_vks = inv[~inv["video_key"].isin(used_vks) & ~inv["source_prefix"].str.match(r"^(youtube-|0000\d{3}_|image\d+_|output|\d+_(jpg|png))")].groupby("video_key").size().sort_values(ascending=False).index.tolist()
sheet10 = pick_from_vkeys(rem_vks[:16], 16)

# Sheet 11: Additional youtube-N_jpg prefixes across the 126 youtube prefixes (e.g. youtube-12..youtube-265)
yt_p = sorted(uninspected[uninspected["source_prefix"].str.match(r"^youtube-\d+_jpg$")]["source_prefix"].unique(),
              key=lambda x: int(re.findall(r"\d+", x)[0]))
yt_step = max(1, len(yt_p) // 16)
sheet11 = pick_from_prefixes([yt_p[i * yt_step] for i in range(min(16, len(yt_p)))], 16, prefer_clean=False)

# Sheet 12: Automated Anomaly / Cutout / Mosaic Detector Verification Sheet (8 suspected noise-cutout boxes + 8 high seam_row/seam_col mosaic candidates outside youtube-1..11)
b_det = boxes[boxes["line_format"] == "DETECTION"].copy()
cut_cands = b_det[(b_det["in_dark"] > 0.80) & (b_det["in_sp"] > 0.01) & (b_det["area"] < 0.85)].merge(
    uninspected, on=["split", "rel_image_path"]).drop_duplicates("source_prefix").head(8)
mos_cands = uninspected[(uninspected["seam_row"] > 6.0) & (uninspected["seam_col"] > 6.0)].drop_duplicates("source_prefix").head(8)
sheet12 = [inv.loc[inv["rel_image_path"] == p].iloc[0] for p in list(cut_cands["rel_image_path"]) + list(mos_cands["rel_image_path"])]

SHEETS = [
    ("phase2_sheet_01_class1_and_kakaotalk.jpg", sheet01),
    ("phase2_sheet_02_output_family.jpg", sheet02),
    ("phase2_sheet_03_drowning_and_clipchamp.jpg", sheet03),
    ("phase2_sheet_04_class_offset_pairs.jpg", sheet04),
    ("phase2_sheet_05_stock_and_2022_videos.jpg", sheet05),
    ("phase2_sheet_06_videoN_and_other_mp4.jpg", sheet06),
    ("phase2_sheet_07_imageNNNN_family.jpg", sheet07),
    ("phase2_sheet_08_pure_digits_family.jpg", sheet08),
    ("phase2_sheet_09_val_train_img_families.jpg", sheet09),
    ("phase2_sheet_10_top_other_static.jpg", sheet10),
    ("phase2_sheet_11_youtube_extended.jpg", sheet11),
    ("phase2_sheet_12_cutout_and_mosaic_audit.jpg", sheet12),
]

COLORS = {
    0: (0, 165, 255),    # orange
    1: (255, 0, 255),    # magenta
    2: (255, 255, 0),    # cyan
    3: (0, 255, 255),    # yellow
    4: (0, 255, 0),      # green
    5: (0, 0, 255),      # red
    6: (255, 128, 0),    # blue
}

manifest_rows = []
for sname, srows in SHEETS:
    tiles = []
    for tidx, r in enumerate(srows, 1):
        sp = r["split"]; rel = r["rel_image_path"]
        data = ZIPS[sp].read(rel)
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (360, 360), interpolation=cv2.INTER_AREA)
        ib = boxes[(boxes["split"] == sp) & (boxes["rel_image_path"] == rel)]
        for _, bx in ib.iterrows():
            if pd.notna(bx["cx"]):
                cid = int(bx["class_id"]) if pd.notna(bx["class_id"]) else 0
                col = COLORS.get(cid, (255, 255, 255))
                x1 = int((bx["cx"] - bx["w"] / 2) * 360)
                y1 = int((bx["cy"] - bx["h"] / 2) * 360)
                x2 = int((bx["cx"] + bx["w"] / 2) * 360)
                y2 = int((bx["cy"] + bx["h"] / 2) * 360)
                cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
                cv2.putText(img, f"c{cid}", (max(2, x1), max(14, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        # Header banner
        banner = np.zeros((42, 360, 3), dtype=np.uint8)
        title1 = f"T{tidx:02d} [{sp}] cls={r['class_ids']} ({len(ib)}b)"
        title2 = f"{r['source_prefix'][:42]}"
        cv2.putText(banner, title1, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(banner, title2, (4, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 230, 255), 1, cv2.LINE_AA)
        tile = np.vstack([banner, img])
        tiles.append(tile)
        manifest_rows.append({
            "sheet": sname,
            "tile_index": tidx,
            "image_id": r["image_id"],
            "split": sp,
            "rel_image_path": rel,
            "source_prefix": r["source_prefix"],
            "video_key": r["video_key"],
            "class_ids": r["class_ids"],
            "n_det": r["n_det"],
        })
    while len(tiles) < 16:
        tiles.append(np.zeros((402, 360, 3), dtype=np.uint8))
    grid = np.vstack([np.hstack(tiles[r * 4 : (r + 1) * 4]) for r in range(4)])
    cv2.imwrite(os.path.join(CS_DIR, sname), grid, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    print("Wrote", sname, len(srows))

pd.DataFrame(manifest_rows).to_csv(os.path.join(CS_DIR, "phase2_contact_sheet_index.csv"), index=False)
print("Total Phase 2 contact-sheet tiles:", len(manifest_rows))

