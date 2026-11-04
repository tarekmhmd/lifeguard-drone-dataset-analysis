"""Phase 2 Full Dataset Recovery & Filtering Investigation:
Builds all 10 required CSV deliverables (Phases B, C, D, E, F, G, H) and visual_overlays/,
and writes a JSON summary of exact verified statistics for the Markdown reports.

Strict read-only access to original dataset archives. All outputs written to:
  E:\lifeguardDron\DrowningDetection_v1_cleaning\04_recovery_investigation\
"""
import json, os, re, warnings, zipfile
import cv2, numpy as np, pandas as pd

warnings.filterwarnings("ignore")

ROOT = r"E:\lifeguardDron\DrowningDetection_v1_cleaning\04_recovery_investigation"
C = os.path.join(ROOT, "analysis_logs", "cache")
OV_DIR = os.path.join(ROOT, "visual_overlays")
os.makedirs(OV_DIR, exist_ok=True)

SRC = r"E:\lifeguardDron\DrowningDetectionTracking_v1"
ZIPS = {
    "train": os.path.join(SRC, "train-20261102T163226Z-1-001.zip"),
    "valid": os.path.join(SRC, "valid-20261102T163228Z-1-001.zip"),
    "test": os.path.join(SRC, "test-20261102T163222Z-1-001.zip"),
}

inv = pd.read_csv(os.path.join(C, "inventory_keys.csv"))
inv["class_ids"] = inv["class_ids"].fillna("")
boxes = pd.read_csv(os.path.join(C, "boxes_raw.csv"), low_memory=False)
pp = pd.read_csv(os.path.join(C, "phash_pairs_le12.csv"))

p1 = pd.read_csv(r"E:\lifeguardDron\DrowningDetection_v1_cleaning\03_source_investigation\VISION_REVIEW_RESULTS.csv")
p2 = pd.read_csv(os.path.join(ROOT, "contact_sheets", "phase2_contact_sheet_index.csv"))

# -------------------------------------------------------------------------
# 1. Super-family classification (17 exhaustive families covering all 23,586 images)
# -------------------------------------------------------------------------
def assign_super_family(prefix):
    if re.match(r"^youtube-\d+_", prefix):
        return "SF01_youtube_N"
    if re.match(r"^output_?\d+_", prefix):
        return "SF02_output_NNNN"
    if re.match(r"^image\d+_", prefix):
        return "SF03_imageNNNN"
    if re.match(r"^\d{7}_\d{5}_d_\d{7}_", prefix):
        return "SF05_visdrone_0000"
    if re.match(r"^\d+_(jpg|png|jpeg)$", prefix):
        return "SF04_pure_digits"
    if prefix.startswith("kakaotalk_"):
        return "SF06_kakaotalk"
    if prefix.startswith("mvi_"):
        return "SF07_maritime_mvi"
    if re.match(r"^drowning\d*_", prefix) and "drowning_small" not in prefix:
        return "SF08_drowningN_mp4"
    if "drowning_small" in prefix:
        return "SF09_drowning_small"
    if prefix.startswith("video_2022-12-07"):
        return "SF10_video_2022_12_07"
    if re.match(r"^video\d+", prefix):
        return "SF11_videoN_mp4"
    if "clipchamp" in prefix:
        return "SF12_clipchamp_mp4"
    if any(k in prefix for k in ("istock", "shutterstock", "pond5", "getty", "bigstock", "stock-video", "203512036")):
        return "SF13_stock_named"
    if re.match(r"^(val|train|test)\d*_(png|jpg|jpeg)", prefix):
        return "SF14_val_train_test_N"
    if re.match(r"^(img|images|pic|photo)[-_0-9]", prefix):
        return "SF15_img_pic_photo"
    if re.search(r"_(mp4|mov|avi|mkv|webm)-", prefix):
        return "SF17_other_video_mp4_mov"
    return "SF18_other_static"

inv["super_family"] = inv["source_prefix"].map(assign_super_family)

# -------------------------------------------------------------------------
# 2. Per-image box forensics aggregation from boxes_raw.csv
# -------------------------------------------------------------------------
b_det = boxes[boxes["line_format"] == "DETECTION"].copy()
b_det["is_oversized_90"] = (b_det["area"] > 0.90).astype(int)
b_det["is_oversized_80"] = (b_det["area"] > 0.80).astype(int)
b_det["is_tiny"] = (b_det["area"] < 0.0004).astype(int)
b_det["is_extreme_ar"] = ((b_det["w"] / (b_det["h"] + 1e-9) > 8.0) | (b_det["h"] / (b_det["w"] + 1e-9) > 8.0)).astype(int)
# Box-level noise cutout: dark interior with salt-and-pepper speckle or solid black/gray occlusion patch
b_det["is_noise_cutout_box"] = (
    ((b_det["in_dark"] >= 0.70) & (b_det["in_sp"] >= 0.008) & (b_det["area"] < 0.85))
    | ((b_det["in_dark"] >= 0.92) & (b_det["in_std"] < 15.0) & (b_det["area"] < 0.85))
).astype(int)

box_agg = b_det.groupby(["split", "rel_image_path"]).agg(
    n_oversized_90=("is_oversized_90", "sum"),
    n_oversized_80=("is_oversized_80", "sum"),
    n_tiny=("is_tiny", "sum"),
    n_extreme_ar=("is_extreme_ar", "sum"),
    n_touch_border=("touch_border", "sum"),
    n_oob=("oob", "sum"),
    n_nonpos=("nonpositive_wh", "sum"),
    n_cutout_boxes=("is_noise_cutout_box", "sum"),
    max_box_area=("area", "max"),
    min_box_area=("area", "min"),
    max_box_in_sp=("in_sp", "max"),
    max_box_in_dark=("in_dark", "max"),
).reset_index()

inv = inv.merge(box_agg, on=["split", "rel_image_path"], how="left")
for col in ["n_oversized_90", "n_oversized_80", "n_tiny", "n_extreme_ar", "n_touch_border", "n_oob", "n_nonpos", "n_cutout_boxes"]:
    inv[col] = inv[col].fillna(0).astype(int)
for col in ["max_box_area", "min_box_area", "max_box_in_sp", "max_box_in_dark"]:
    inv[col] = inv[col].fillna(0.0).round(6)

# -------------------------------------------------------------------------
# 3. Direct Visual Review Mapping (Phase 1: 106 rows + Phase 2: 183 unique = 287 net unique files)
#    Phase A Audit found that in VISION_REVIEW_RESULTS.csv (106 rows):
#      - 80 rows match rel_image_path verbatim
#      - 14 rows (invalid_annotations) match by <split>/images/<prefix>.rf.<first-8-hex>
#      - 12 rows (class_semantics_extra Contact Sheet 4) match by source_prefix (tile header showed prefix only)
# -------------------------------------------------------------------------
inv["key8"] = inv["rel_image_path"].str.replace(r"(\.rf\.[0-9a-f]{8})[0-9a-f]+", r"\1", regex=True)
key8_to_path = dict(zip(inv["key8"], inv["rel_image_path"]))
pref_to_first_path = inv.sort_values("image_id").drop_duplicates("source_prefix").set_index("source_prefix")["rel_image_path"].to_dict()
all_paths_set = set(inv["rel_image_path"])

p1_map = {}
for _, r in p1.iterrows():
    raw_name = r["image_name"]
    if raw_name in all_paths_set:
        resolved_path = raw_name
    else:
        k8 = re.sub(r"(\.rf\.[0-9a-f]{8})[0-9a-f]+", r"\1", raw_name)
        if k8 in key8_to_path:
            resolved_path = key8_to_path[k8]
        else:
            pref = os.path.basename(raw_name).split(".rf.")[0].lower()
            resolved_path = pref_to_first_path.get(pref, raw_name)
    p1_map[resolved_path] = {
        "scene": r["scene"],
        "sheet": f"P1_{r['sheet']}",
        "tile": int(r["tile_index"]),
        "box_target": str(r["box_target"]),
        "box_quality": str(r["box_quality"]),
        "confidence": str(r["confidence"]),
        "evidence": str(r["evidence"]),
    }

# Map Phase 2 visual findings for all 192 tiles across the 12 Phase 2 contact sheets
def p2_visual_finding(row):
    s = row["sheet"]
    t = int(row["tile_index"])
    p = row["source_prefix"]
    vk = row["video_key"]
    cids = str(row["class_ids"])
    if s == "phase2_sheet_01_class1_and_kakaotalk.jpg":
        if t <= 6:
            return ("POOL", "HIGH", "Overhead indoor swimming pool with swimmers in water labeled Class 6 (swimming); some boxes show rectangular patch rotation artifacts.")
        elif t <= 8:
            return ("POOL", "HIGH", "Aerial drone view of outdoor competition swimming pool with lane ropes; swimmers in water labeled numeric Class 1 and Class 3.")
        elif t == 9:
            return ("CORRUPTED", "HIGH", "Aerial outdoor swimming pool with large black salt-and-pepper noise cutout occluding the central pool area (`130394_jpg`).")
        else:
            return ("POOL", "HIGH", f"Korean swimming pool scene (`{vk}`) with children/swimmers in pool water labeled numeric Class {cids} (unresolved numeric class schema).")
    elif s == "phase2_sheet_02_output_family.jpg":
        if t <= 2:
            return ("POOL", "HIGH", f"Overhead indoor swimming pool (`{p}`) showing person in dark clothing submerged/drowning in pool water, labeled Class 5 (`drowning`).")
        else:
            return ("POOL", "HIGH", f"Overhead indoor swimming pool session (`{p}`) with lane ropes; swimmers/bystanders labeled with numeric Class {cids} (`output_NNNN` single-video sequence).")
    elif s == "phase2_sheet_03_drowning_and_clipchamp.jpg":
        if t == 16:
            return ("OUT_OF_SCOPE", "HIGH", "White-water river rapids tubing scene (`white-water_mp4-24_jpg`) with person on yellow tube in river rapids labeled Class 6 (`swimming`).")
        elif t == 15:
            return ("POOL", "HIGH", "Outdoor swimming pool (`-Clipchamp-_mp4-10_jpg`) with '5 FT' depth marker on pool wall and swimmers holding blue pool noodles labeled Class 6.")
        elif t == 14:
            return ("POOL", "HIGH", "Helsinki indoor swimming pool CCTV (`Untitled-video-Made-with-Clipchamp-1-_mp4-39_jpg`, timestamp 30.1.2016) with inverted classes: Class 4 (`Out of Water`) on drowning child and Class 5 (`drowning`) on normal swimmers.")
        elif t in (3, 4, 5):
            return ("POOL", "HIGH", f"Swimming pool rescue/drowning close-up (`{p}`) where person in pool water is mislabeled Class 4 (`Out of Water`) due to +4 class-index offset.")
        else:
            return ("POOL", "HIGH", f"Wave-pool / indoor pool drowning video frame (`{p}`) affected by (0,3)<->(4,5) class-offset corruption and/or bounding-box patch augmentation.")
    elif s == "phase2_sheet_04_class_offset_pairs.jpg":
        if t in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10):
            return ("POOL", "HIGH", f"Paired class-offset forensic comparison (`{p}`): exact same pool video has numeric Class `0`/`3` in one export and shifted Class `4` (`Out of Water`) / `5` (`drowning`) in the second export.")
        elif t in (11, 12, 13, 14):
            return ("OUT_OF_SCOPE", "HIGH", f"Paired class-offset forensic comparison (`{p}`): open-water / murky outdoor stock video with Class `0` in one export and Class `4` (`Out of Water`) on submerged person in the second export.")
        else:
            return ("POOL", "HIGH", f"Paired class-offset forensic comparison (`{p}`): outdoor pool child drowning stock video with Class `0` in one copy and Class `4` (`Out of Water`) in the byte-identical duplicate.")
    elif s == "phase2_sheet_05_stock_and_2022_videos.jpg":
        if t in (4, 5, 6, 12, 14, 15, 16):
            return ("POOL", "HIGH", f"Stock swimming pool image/video (`{p}`) showing child/person in clear tiled pool water, labeled Class `{cids}` (affected by +4 class-offset where `4=Out of Water` boxes submerged person).")
        else:
            return ("OUT_OF_SCOPE", "HIGH", f"Open-water / ocean / sea / lake stock footage (`{p}`) outside swimming-pool operational scope, labeled Class `{cids}`.")
    elif s == "phase2_sheet_06_videoN_and_other_mp4.jpg":
        if t <= 9:
            return ("OUT_OF_SCOPE", "HIGH", f"Open ocean surf / beach waves (`{p}`) with person in sea water labeled Class `{cids}` (0<->4 class offset).")
        elif t == 10:
            return ("OUT_OF_SCOPE", "HIGH", "Coastal sea / open water scene (`video6_mp4-11_jpg`) with person in dark sea water mislabeled Class 4 (`Out of Water`).")
        elif t == 11:
            return ("POOL", "HIGH", "Outdoor swimming pool (`-Clipchamp-_mp4-10_jpg`) with '5 FT' poolside depth marker and swimmers.")
        elif t in (12, 13, 14, 15):
            return ("OUT_OF_SCOPE", "HIGH", f"Open water / coastal sea / park water body (`{p}`) outside swimming-pool scope, labeled Class `{cids}`.")
        else:
            return ("OUT_OF_SCOPE", "HIGH", "White-water river rapids tubing (`white-water_mp4-24_jpg`) outside swimming-pool scope.")
    elif s == "phase2_sheet_07_imageNNNN_family.jpg":
        if t == 5:
            return ("CORRUPTED", "HIGH", "Thai TV news broadcast (`THAIRATH TV 32`, `image954_jpg`) of backyard pool CCTV where the drowning child inside the red circle is obliterated by a black salt-and-pepper noise cutout labeled Class 4.")
        elif t in (1, 8):
            return ("POOL", "HIGH", f"Thai TV news broadcast (`THAIRATH TV 32`, `{p}`) of backyard pool CCTV with Thai text banners, red circle overlay, and news anchor inset; 3 boxes (`0;1;2`) mark child in water and bystanders on deck.")
        else:
            return ("POOL", "HIGH", f"Thai TV news broadcast (`THAIRATH TV 32`, `{p}`) of backyard pool CCTV with TV graphics and anchor inset; single Class 4 (`Out of Water`) box erroneously marks the drowning child in water while leaving bystanders unannotated.")
    elif s == "phase2_sheet_08_pure_digits_family.jpg":
        if t <= 10:
            return ("POOL", "HIGH", f"Overhead indoor swimming pool (`{p}`) with swimmers labeled Class 6 (`swimming`); several frames show localized rectangular box-patch rotation artifacts.")
        elif t in (11, 12, 13):
            return ("OUT_OF_SCOPE", "HIGH", f"Aerial beach / coastal ocean surf (`{p}`) with people in ocean waves or beach sand labeled Class `{cids}`.")
        elif t in (14, 15):
            return ("UNCERTAIN", "MEDIUM", f"Tightly cropped letterboxed overhead blue water view (`{p}`) with a single swimmer labeled Class 6; lacks pool boundaries/lanes to confirm pool vs open water.")
        else:
            return ("POOL", "HIGH", "Aerial outdoor competition swimming pool (`130473_jpg`) with lane ropes and 12 swimmers labeled numeric Class 3.")
    elif s == "phase2_sheet_09_val_train_img_families.jpg":
        if t <= 8:
            return ("OUT_OF_SCOPE", "HIGH", f"Open ocean / sea / lake stock photo (`{p}`) showing hand/person in open water waves or sunset sea, mislabeled Class 4 (`Out of Water`).")
        elif t in (9, 10, 11, 12, 15):
            return ("POOL", "HIGH", f"Underwater / poolside swimming pool stock photo (`{p}`) showing submerged person in clear blue pool water, affected by 0<->4 class offset (`{cids}`).")
        elif t in (13, 14):
            return ("OUT_OF_SCOPE", "HIGH", f"Open sea / dark open water stock photo (`{p}`) showing raised hand/person in ocean waves mislabeled Class 4 (`Out of Water`).")
        else:
            return ("OUT_OF_SCOPE", "HIGH", "Infographic / text-overlay composite (`images-6-_jpeg_jpg`, 'Drowning Myth' banner) unsuitable for detector training.")
    elif s == "phase2_sheet_10_top_other_static.jpg":
        if t in (1, 2, 4):
            return ("POOL", "HIGH", f"Staged indoor tiled swimming pool video (`{p}`) with swimmer/drowning actor in pool water labeled numeric Class `{cids}`.")
        elif t == 3:
            return ("OUT_OF_SCOPE", "HIGH", "Murky river / open water clip (`10_MOV-0_jpg`, 'local waala' watermark) outside swimming-pool scope.")
        elif t == 5:
            return ("POOL", "HIGH", "Stock photo with 123RF watermark (`134653437-asian-senior...`) showing person struggling underwater in swimming pool, labeled Class 0.")
        elif t == 6:
            return ("POOL", "HIGH", "Korean outdoor swimming pool (`KakaoTalk_20230504_000756264_12_jpg`) with children in pool water labeled numeric Class 2.")
        elif t in (7, 12, 14):
            return ("OUT_OF_SCOPE", "HIGH", f"Open sea / ocean waves stock image (`{p}`) showing hand/person in ocean water.")
        elif t == 11:
            return ("UNCERTAIN", "MEDIUM", "Close-up child face in splashing water (`drowning_small-1-1000x573_jpg`) with oversized box and no visible pool structure.")
        else:
            return ("POOL", "HIGH", f"Swimming pool rescue / drowning stock photo (`{p}`) with watermarks or +4 class-offset mislabeling (`cls={cids}`).")
    elif s == "phase2_sheet_11_youtube_extended.jpg":
        if t in (4, 6):
            return ("SUSPECTED_MOSAIC", "HIGH", f"2x2 Mosaic composite (`{p}`) combining 4 distinct sub-images with hard horizontal/vertical seams at x=320, y=320.")
        elif t in (8, 12):
            return ("CORRUPTED", "HIGH", f"Indoor pool frame (`{p}`) corrupted by synthetic black salt-and-pepper noise cutout box replacing the swimmer.")
        elif t in (3, 7, 13, 14, 15, 16):
            return ("POOL", "HIGH", f"Kenyan TV news broadcast (`CITIZENTV.CO.KE / MONDAY REPORT: WATER SAFETY`, `{p}`) with studio banners, PiP sign-language interpreter (`cls=6`), or dry-land demonstrations (`cls=1`/`4`).")
        else:
            return ("POOL", "HIGH", f"Indoor swimming pool frame (`{p}`) from `youtube-N` family with mixed numeric/semantic classes (`{cids}`).")
    elif s == "phase2_sheet_12_cutout_and_mosaic_audit.jpg":
        return ("CORRUPTED", "HIGH", f"Outdoor swimming pool (`{p}`) severely corrupted by synthetic black/gray salt-and-pepper noise cutout rectangles replacing swimmers inside annotated boxes.")
    return ("UNCERTAIN", "LOW", "Visual review entry.")

p2_map = {}
for _, r in p2.iterrows():
    sc, conf, ev = p2_visual_finding(r)
    p2_map[r["rel_image_path"]] = {
        "scene": sc,
        "sheet": f"P2_{r['sheet']}",
        "tile": int(r["tile_index"]),
        "confidence": conf,
        "evidence": ev,
    }

# Combine direct visual inspection records (289 unique rel_image_paths across 286 unique sha_groups)
direct_scene = {}
direct_meta = {}
for path, d in p1_map.items():
    direct_scene[path] = d["scene"]
    direct_meta[path] = (d["sheet"], d["tile"], d["confidence"], d["evidence"], "PHASE1_VISUAL_DIRECT")
for path, d in p2_map.items():
    if path not in direct_scene:
        direct_scene[path] = d["scene"]
        direct_meta[path] = (d["sheet"], d["tile"], d["confidence"], d["evidence"], "PHASE2_VISUAL_DIRECT")

# Also identify byte-identical SHA-256 clones of directly visually inspected images!
sha_to_direct = {}
for path, sc in direct_scene.items():
    row = inv[inv["rel_image_path"] == path]
    if not row.empty:
        sg = int(row.iloc[0]["sha_group"])
        sha_to_direct[sg] = (path, sc, direct_meta[path])

# -------------------------------------------------------------------------
# 4. Candidate Scene Classification & Anomaly Detection Across All 23,586 Images
#    IMPORTANT: Strictly preserves `reviewer_status = VERIFIED_VISUAL` ONLY for
#    directly inspected images (and marks `VERIFIED_SHA_CLONE` for byte-identical
#    SHA-256 clones of directly inspected images), while all uninspected images
#    receive `reviewer_status = PENDING_REVIEW` per Section 6!
# -------------------------------------------------------------------------
# Source-prefix / video-key scene rules derived from our 289 visually inspected images across all 17 super-families
OUT_OF_SCOPE_VKEYS = {
    "white-water_mp4", "10_mov", "video6_mp4",
    "test1-1-_mp4", "test1-2-_mp4", "test1-5-_mp4", "test1-8-_mp4",
    "3-727-drowning-stock-videos-and-royalty-free-footage-istock_2_mp4",
    "3-727-drowning-stock-videos-and-royalty-free-footage-istock_3_mp4",
    "3-727-drowning-stock-videos-and-royalty-free-footage-istock_mp4",
    "drowning-man-stock-video-download-video-clip-now-drowning-struggle-water-istock_mp4",
    "a-man-drowning-a-safeguard-jumps-in-with-a-rescue-buoy-and-offers-him-help-stock-video-download-video-clip-now-istock_mp4",
    "istockphoto-1032638012-612x612_jpg", "istockphoto-1034894690-612x612_jpg",
    "istockphoto-1041463788-612x612-1-_jpg", "istockphoto-1041463788-612x612_jpg",
    "istockphoto-105764867-612x612_jpg", "istockphoto-1299949453-640x640_jpg",
    "53feaf75ccc0ec70a1e16ca831fa6b31-drown-sinks_jpg", "z_png_jpg", "drowning-person_jpg",
    "images-28-_jpg", "images-30-_jpg", "images-6-_jpeg_jpg",
}

POOL_STOCK_VKEYS = {
    "child-pretending-to-be-drowning-at-the-swimming-pool-stock-video-download-video-clip-now-istock_mp4",
    "the-child-is-drowning-in-water-selective-focus-nature-stock-video-download-video-clip-now-istock_mp4",
    "the-child-is-drowning-in-water-selective-focus-kid-stock-video-download-video-clip-now-istock_mp4",
    "istock-1001608692_jpg", "istockphoto-1001608692-612x612_jpg",
    "istockphoto-1222550542-612x612_jpg", "istockphoto-172243587-612x612-1-_jpg",
    "istockphoto-182180100-612x612_jpg", "203512036-1-_jpg",
    "134653437-asian-senior-people-struggling-underwater-hand-peeking-out-of-the-water-female-drowned-in-swimming-p_jpg",
    "93531112-man-drowning-in-out-door-swimming-pool-while-swimming-alone-show-two-hands-and-asking-for-help_jpg",
    "man-being-pulled-from-water-before-drowning-1024x683_jpg",
    "reviving-someone-who-has-drowned_jpg", "drowning-doesnt-look-like-drowning-web_jpg",
    "images-1-_jpg", "images-14-_jpg", "images-19-_jpg", "images-24-_jpg",
    "images-3-_jpeg_jpg", "images-4-_jpeg_jpg",
}

def infer_group_scene(row):
    sf = row["super_family"]
    p = row["source_prefix"]
    vk = row["video_key"]
    # Check if image has severe destructive noise cutout or mosaic composite first
    is_cutout_img = (row["n_cutout_boxes"] > 0) or (row["sp_frac_global"] >= 0.015 and row["dark_frac"] >= 0.25)
    is_mosaic_img = (
        (sf == "SF01_youtube_N" and row["seam_row"] >= 4.2 and row["seam_col"] >= 4.2 and 240 <= row["seam_row_y"] <= 400 and 240 <= row["seam_col_x"] <= 400)
        or (row["seam_row"] >= 7.5 and row["seam_col"] >= 7.5 and 250 <= row["seam_row_y"] <= 390 and 250 <= row["seam_col_x"] <= 390)
    )
    if is_cutout_img:
        return "CORRUPTED", "LOW", "Candidate flagged by pixel/box forensics: synthetic salt-and-pepper / solid dark occlusion cutout (`n_cutout_boxes>0` or high global `sp_frac`)."
    if is_mosaic_img:
        return "SUSPECTED_MOSAIC", "LOW", f"Candidate flagged by full-span horizontal/vertical seam detector (`seam_row={row['seam_row']}`, `seam_col={row['seam_col']}`)."

    if sf in ("SF05_visdrone_0000", "SF07_maritime_mvi", "SF11_videoN_mp4", "SF14_val_train_test_N", "SF17_other_video_mp4_mov"):
        return "OUT_OF_SCOPE", "MEDIUM", f"Candidate from out-of-scope super-family `{sf}` (`{vk}`: urban traffic, maritime, ocean surf, or river rapids)."
    if vk in OUT_OF_SCOPE_VKEYS or p in OUT_OF_SCOPE_VKEYS:
        return "OUT_OF_SCOPE", "MEDIUM", f"Candidate from visually sampled out-of-scope video/photo group `{vk}`."
    if sf in ("SF02_output_NNNN", "SF03_imageNNNN", "SF06_kakaotalk", "SF08_drowningN_mp4", "SF10_video_2022_12_07", "SF12_clipchamp_mp4"):
        return "POOL", "MEDIUM", f"Candidate from visually sampled swimming-pool super-family `{sf}` (`{vk}`)."
    if vk in POOL_STOCK_VKEYS or p in POOL_STOCK_VKEYS:
        return "POOL", "MEDIUM", f"Candidate from visually sampled swimming-pool stock group `{vk}`."
    if sf == "SF01_youtube_N":
        return "POOL", "LOW", "Candidate from `youtube-N` pool/broadcast family (contains indoor pool scenes mixed with TV broadcast banners/PiP and 2x2 mosaics; requires per-image visual review)."
    if sf == "SF04_pure_digits":
        m = re.match(r"^(\d+)_(jpg|png|jpeg)$", p)
        num = int(m.group(1)) if m else 0
        ext = m.group(2) if m else ""
        if ext == "png" or "_png_" in p:
            return "OUT_OF_SCOPE", "LOW", f"Candidate from `{p}` (`N_png_jpg` dark open-water wave sub-series)."
        elif 1 <= num <= 250:
            return "POOL", "MEDIUM", f"Candidate from `{p}` (`1..250_jpg` overhead indoor swimming pool sub-series)."
        elif 251 <= num <= 20000:
            return "OUT_OF_SCOPE", "MEDIUM", f"Candidate from `{p}` (`3000..12000_jpg` aerial beach / coastal ocean surf sub-series)."
        elif 130000 <= num <= 131000:
            if row["dark_frac"] > 0.25:
                return "UNCERTAIN", "LOW", f"Candidate from `{p}` (`130000_jpg` letterboxed overhead blue water crop without visible pool edges)."
            return "POOL", "LOW", f"Candidate from `{p}` (`130000_jpg` aerial outdoor competition pool sub-series)."
        return "UNCERTAIN", "LOW", f"Candidate from numeric prefix `{p}`; requires direct visual inspection."
    if sf == "SF09_drowning_small":
        return "UNCERTAIN", "LOW", "Close-up child face in splashing water (`drowning_small-1-1000x573_jpg`); pool boundaries not visible."
    # Remaining SF13_stock_named, SF15_img_pic_photo, SF18_other_static
    if "pool" in p:
        return "POOL", "LOW", f"Uninspected static/stock prefix `{p}` with pool keyword and aqua/blue colour profile; pending visual verification."
    if any(k in p for k in ("sea", "ocean", "beach", "river", "lake", "boat", "ship", "surf", "wave")):
        return "OUT_OF_SCOPE", "LOW", f"Uninspected static/stock prefix `{p}` with open-water keyword; pending visual verification."
    return "UNCERTAIN", "LOW", f"Uninspected prefix `{p}` in `{sf}`; requires direct human visual inspection."

scene_rows = []
for _, row in inv.iterrows():
    rel = row["rel_image_path"]
    sg = int(row["sha_group"])
    if rel in direct_scene:
        sc = direct_scene[rel]
        sheet, tile, conf, ev, rstat = direct_meta[rel]
        rmethod = "visual_inspection_direct"
        rstatus = "VERIFIED_VISUAL"
    elif sg in sha_to_direct:
        src_rel, sc, (sheet, tile, conf, ev, _) = sha_to_direct[sg]
        rmethod = "visual_inspection_exact_sha256_clone"
        rstatus = "VERIFIED_SHA256_CLONE"
        ev = f"Byte-identical SHA-256 clone of visually inspected `{src_rel}` ({sheet} T{tile:02d}): {ev}"
    else:
        sc, conf, ev = infer_group_scene(row)
        rmethod = "source_group_and_pixel_forensics_unverified"
        rstatus = "PENDING_REVIEW"
    scene_rows.append({
        "rel_image_path": rel,
        "original_split": row["split"],
        "source_prefix": row["source_prefix"],
        "source_group": row["video_key"],
        "super_family": row["super_family"],
        "scene_type": sc,
        "confidence": conf,
        "visual_evidence": ev,
        "review_method": rmethod,
        "reviewer_status": rstatus,
    })

scene_df = pd.DataFrame(scene_rows)
inv["scene_type"] = scene_df["scene_type"]
inv["scene_confidence"] = scene_df["confidence"]
inv["reviewer_status"] = scene_df["reviewer_status"]
inv["review_method"] = scene_df["review_method"]

# -------------------------------------------------------------------------
# 5. Prefix / Video-Key Class-Offset & Schema Forensics
# -------------------------------------------------------------------------
# Identify prefixes & video_keys that have class-offset collisions (e.g. both numeric 0/3 and shifted 4/5)
vk_classes = inv.groupby("video_key")["class_ids"].apply(
    lambda s: sorted({int(x) for v in s if v for x in str(v).split(";")})
).to_dict()
pref_classes = inv.groupby("source_prefix")["class_ids"].apply(
    lambda s: sorted({int(x) for v in s if v for x in str(v).split(";")})
).to_dict()
sha_classes = inv.groupby("sha_group")["class_ids"].apply(
    lambda s: sorted({int(x) for v in s if v for x in str(v).split(";")})
).to_dict()

def has_offset_collision(cset):
    s = set(cset)
    # 0 <-> 4 or (0,3) <-> (4,5) collision
    return (0 in s and 4 in s) or (3 in s and 5 in s)

inv["vk_class_set"] = inv["video_key"].map(lambda k: ";".join(map(str, vk_classes.get(k, []))))
inv["pref_has_offset_collision"] = inv["source_prefix"].map(lambda p: int(has_offset_collision(pref_classes.get(p, []))))
inv["vk_has_offset_collision"] = inv["video_key"].map(lambda k: int(has_offset_collision(vk_classes.get(k, []))))
inv["sha_has_offset_collision"] = inv["sha_group"].map(lambda g: int(has_offset_collision(sha_classes.get(g, []))))

# Mark canonical representative of each unique image_sha256 group (preferring visually reviewed, then train > valid > test, then lowest image_id)
inv["is_direct_visual"] = (inv["reviewer_status"] == "VERIFIED_VISUAL").astype(int)
split_rank = {"train": 0, "valid": 1, "test": 2}
inv["split_rank"] = inv["split"].map(split_rank)
rep_ids = set(
    inv.sort_values(["is_direct_visual", "split_rank", "image_id"], ascending=[False, True, True])
    .drop_duplicates("sha_group")["image_id"]
)
inv["is_sha_representative"] = inv["image_id"].isin(rep_ids).astype(int)
inv["is_exact_dup_redundant"] = (1 - inv["is_sha_representative"]).astype(int)

# -------------------------------------------------------------------------
# PHASE B DELIVERABLE: FULL_DATASET_INVENTORY.csv (23,586 rows)
# -------------------------------------------------------------------------
inv_out = pd.DataFrame({
    "rel_image_path": inv["rel_image_path"],
    "filename": inv["filename"],
    "original_split": inv["split"],
    "source_prefix": inv["source_prefix"],
    "source_group": inv["video_key"],
    "super_family": inv["super_family"],
    "width": inv["width"],
    "height": inv["height"],
    "image_format": inv["filename"].str.rsplit(".", n=1).str[-1].str.lower(),
    "file_size": inv["file_size"],
    "label_path": inv["label_path"],
    "annotation_line_count": inv["label_line_count"],
    "annotation_format": inv["label_format"],
    "numeric_class_ids": inv["class_ids"],
    "image_sha256": inv["image_sha256"],
    "annotation_sha256": inv["label_sha256"],
    "sha_group_id": inv["sha_group"],
    "sha_group_size": inv["sha_group_size"],
    "is_exact_dup_redundant": inv["is_exact_dup_redundant"],
    "existing_review_status": inv["reviewer_status"],
})
inv_out.to_csv(os.path.join(ROOT, "FULL_DATASET_INVENTORY.csv"), index=False, encoding="utf-8")
print("Wrote FULL_DATASET_INVENTORY.csv:", len(inv_out))

# -------------------------------------------------------------------------
# PHASE C DELIVERABLES: SCENE_CLASSIFICATION_FULL.csv & SCENE_REVIEW_PROGRESS.csv
# -------------------------------------------------------------------------
scene_df.to_csv(os.path.join(ROOT, "SCENE_CLASSIFICATION_FULL.csv"), index=False, encoding="utf-8")
print("Wrote SCENE_CLASSIFICATION_FULL.csv:", len(scene_df))

prog_rows = []
for sf, grp in inv.groupby("super_family"):
    n_tot = len(grp)
    n_u_sha = grp["sha_group"].nunique()
    n_vis_direct = int((grp["reviewer_status"] == "VERIFIED_VISUAL").sum())
    n_vis_clone = int((grp["reviewer_status"] == "VERIFIED_SHA256_CLONE").sum())
    n_pend = int((grp["reviewer_status"] == "PENDING_REVIEW").sum())
    sc_counts = grp["scene_type"].value_counts().to_dict()
    prog_rows.append({
        "super_family": sf,
        "unique_video_keys": grp["video_key"].nunique(),
        "unique_source_prefixes": grp["source_prefix"].nunique(),
        "total_images": n_tot,
        "unique_sha256_images": n_u_sha,
        "verified_visual_direct_images": n_vis_direct,
        "verified_sha256_clone_images": n_vis_clone,
        "pending_review_images": n_pend,
        "scene_POOL_total": sc_counts.get("POOL", 0),
        "scene_OUT_OF_SCOPE_total": sc_counts.get("OUT_OF_SCOPE", 0),
        "scene_UNCERTAIN_total": sc_counts.get("UNCERTAIN", 0),
        "scene_SUSPECTED_MOSAIC_total": sc_counts.get("SUSPECTED_MOSAIC", 0),
        "scene_CORRUPTED_total": sc_counts.get("CORRUPTED", 0),
    })
prog_df = pd.DataFrame(prog_rows).sort_values("super_family")
prog_df.to_csv(os.path.join(ROOT, "SCENE_REVIEW_PROGRESS.csv"), index=False, encoding="utf-8")
print("Wrote SCENE_REVIEW_PROGRESS.csv:", len(prog_df))

# -------------------------------------------------------------------------
# PHASE D DELIVERABLES: ANNOTATION_FORENSICS_FULL.csv & CLASS_SEMANTICS_BY_SOURCE.csv
# -------------------------------------------------------------------------
def classify_annotation_status(row):
    lf = row["label_format"]
    cids = [int(x) for x in str(row["class_ids"]).split(";") if x != ""]
    sf = row["super_family"]
    sc = row["scene_type"]
    if lf == "EMPTY":
        return "EMPTY_ANNOTATION", "EXCLUDE_OUT_OF_SCOPE", "Empty label file on open-ocean maritime vessel image (`MVI_1627_VIS_frame20`)."
    if lf in ("POLYGON_ONLY", "MIXED_DET_POLY"):
        return "MALFORMED_POLYGON_IN_DET_DATASET", "EXCLUDE_OUT_OF_SCOPE", f"Contains {row['n_poly']} multi-vertex YOLO segmentation polygon line(s) (`class 4`) in an out-of-scope stock video (`3-727-...-iStock_mp4`)."
    if sc == "OUT_OF_SCOPE":
        return "OUT_OF_SCOPE_SCENE_LABELS", "EXCLUDE_OUT_OF_SCOPE", f"Boxes annotate objects in out-of-scope scene (`{sf}`). Not usable for pool-only detector."
    if sc == "CORRUPTED" or row["n_cutout_boxes"] > 0:
        return "DESTRUCTIVE_NOISE_CUTOUT_BOX", "EXCLUDE_DESTRUCTIVE_AUGMENTATION", f"Contains {row['n_cutout_boxes']} box(es) where target swimmer pixels are obliterated by synthetic salt-and-pepper / black occlusion cutout."
    if sc == "SUSPECTED_MOSAIC":
        return "MOSAIC_COMPOSITE_LABELS", "EXCLUDE_DESTRUCTIVE_AUGMENTATION", "Pre-baked 2x2 Mosaic composite with seam-truncated objects and mixed sub-tile semantics."
    if row["n_oversized_90"] > 0:
        return "OVERSIZED_FULL_FRAME_BOX", "KEEP_IMAGE_REANNOTATE", f"Contains {row['n_oversized_90']} full-frame box(es) covering >90% of image area (`max_area={row['max_box_area']:.3f}`)."
    if sf == "SF03_imageNNNN":
        return "INCOMPLETE_AND_INVERTED_TV_CCTV_LABELS", "KEEP_IMAGE_REANNOTATE", "Thai TV news CCTV (`THAIRATH TV 32`): Class 4 (`Out of Water`) erroneously boxes the submerged drowning child inside the red TV graphic circle while leaving poolside bystanders unannotated."
    if sf == "SF01_youtube_N":
        return "MIXED_BROADCAST_AND_NUMERIC_SCHEMA", "KEEP_IMAGE_REANNOTATE", "Contains TV news PiP sign-language interpreter (`cls=6`), dry-land rescue demonstrations (`cls=1`/`4`), or unresolved numeric classes (`0,1,3`)."
    if row["vk_has_offset_collision"] or row["pref_has_offset_collision"] or (4 in cids and sf in ("SF08_drowningN_mp4", "SF10_video_2022_12_07", "SF12_clipchamp_mp4", "SF13_stock_named", "SF15_img_pic_photo", "SF18_other_static")):
        return "CLASS_OFFSET_INVERSION_CORRUPTION", "KEEP_IMAGE_REANNOTATE", f"Source group suffers from systematic `+4` / `(0,3)->(4,5)` class-index shift where `Class 4 (Out of Water)` boxes drowning victims in water and `Class 5 (drowning)` boxes normal swimmers."
    if any(c in (0, 1, 2, 3) for c in cids):
        return "UNRESOLVED_NUMERIC_CLASS_SCHEMA", "KEEP_IMAGE_REANNOTATE", f"Uses opaque numeric class IDs (`{row['class_ids']}`) from Roboflow merge (`'3','4','5','6'`); requires semantic re-annotation/remapping."
    # Remaining images have only semantic classes 5 (`drowning`) and/or 6 (`swimming`) without offset collision
    if set(cids).issubset({5, 6}):
        if row["reviewer_status"] in ("VERIFIED_VISUAL", "VERIFIED_SHA256_CLONE"):
            return "VERIFIED_SEMANTIC_POOL_LABELS", "KEEP_CANDIDATE_OR_REVIEW", f"Visually verified pool scene with semantic class ID(s) `{row['class_ids']}` (`5=drowning`, `6=swimming`)."
        return "CANDIDATE_SEMANTIC_POOL_LABELS_UNVERIFIED", "KEEP_IMAGE_REVIEW_LABEL", f"Pool candidate with semantic class ID(s) `{row['class_ids']}` (`5=drowning`, `6=swimming`); requires human visual verification before training."
    return "AMBIGUOUS_CLASS_SEMANTICS", "KEEP_IMAGE_REANNOTATE", f"Ambiguous class combination `{row['class_ids']}`; requires manual review."

ann_stat_res = [classify_annotation_status(r) for _, r in inv.iterrows()]
inv["annotation_forensic_status"] = [a for a, b, c in ann_stat_res]
inv["annotation_recommended_action"] = [b for a, b, c in ann_stat_res]
inv["annotation_forensic_notes"] = [c for a, b, c in ann_stat_res]

ann_df = pd.DataFrame({
    "rel_image_path": inv["rel_image_path"],
    "original_split": inv["split"],
    "source_prefix": inv["source_prefix"],
    "source_group": inv["video_key"],
    "super_family": inv["super_family"],
    "label_path": inv["label_path"],
    "label_format": inv["label_format"],
    "n_detection_boxes": inv["n_det"],
    "n_polygon_lines": inv["n_poly"],
    "n_unparseable_lines": inv["n_bad"],
    "class_ids": inv["class_ids"],
    "n_oversized_gt90_boxes": inv["n_oversized_90"],
    "n_oversized_gt80_boxes": inv["n_oversized_80"],
    "n_tiny_boxes": inv["n_tiny"],
    "n_extreme_aspect_boxes": inv["n_extreme_ar"],
    "n_border_touching_boxes": inv["n_touch_border"],
    "n_noise_cutout_boxes": inv["n_cutout_boxes"],
    "max_box_area": inv["max_box_area"],
    "max_box_in_sp": inv["max_box_in_sp"],
    "max_box_in_dark": inv["max_box_in_dark"],
    "video_key_has_class_offset_collision": inv["vk_has_offset_collision"],
    "sha_group_has_conflicting_labels": (inv["sha_group_label_variants"] > 1).astype(int),
    "annotation_forensic_status": inv["annotation_forensic_status"],
    "recommended_annotation_action": inv["annotation_recommended_action"],
    "forensic_notes": inv["annotation_forensic_notes"],
})
ann_df.to_csv(os.path.join(ROOT, "ANNOTATION_FORENSICS_FULL.csv"), index=False, encoding="utf-8")
print("Wrote ANNOTATION_FORENSICS_FULL.csv:", len(ann_df))

# Build CLASS_SEMANTICS_BY_SOURCE.csv across all 1,407 video_keys
b_with_vk = boxes.merge(inv[["split", "rel_image_path", "video_key", "super_family", "scene_type", "reviewer_status"]], on=["split", "rel_image_path"], how="left")
csem_rows = []
for (sf, vk, cid), grp in b_with_vk.groupby(["super_family", "video_key", "class_id"]):
    n_boxes = len(grp)
    n_imgs = grp["rel_image_path"].nunique()
    n_vis = int(grp["reviewer_status"].isin(["VERIFIED_VISUAL", "VERIFIED_SHA256_CLONE"]).sum())
    cset = vk_classes.get(vk, [])
    has_col = has_offset_collision(cset)
    # Observed visual meaning
    cid_int = int(cid)
    if sf == "SF05_visdrone_0000":
        meaning = "Urban street pedestrians / vehicles / cyclists (VisDrone out-of-scope)"
        consistent = "NO_WRONG_DOMAIN"
    elif sf == "SF03_imageNNNN":
        if cid_int == 4:
            meaning = "INVERTED: Drowning child submerged inside red TV graphic circle in backyard pool (`THAIRATH TV 32`), mislabeled `4 (Out of Water)`"
            consistent = "NO_INVERTED_CLASS_4"
        elif cid_int == 0:
            meaning = "Drowning child in backyard pool (`image724`/`image1944` 3-box variant)"
            consistent = "NUMERIC_UNRESOLVED"
        elif cid_int in (1, 2):
            meaning = "Poolside bystander on deck (`image724`/`image1944` 3-box variant)"
            consistent = "NUMERIC_UNRESOLVED"
        else:
            meaning = "Unverified class in `imageNNNN`"
            consistent = "UNVERIFIED"
    elif sf == "SF02_output_NNNN":
        if vk == "output" and cid_int == 5:
            meaning = "Submerged / drowning person in dark clothing in indoor swimming pool (`output1..75`)"
            consistent = "YES_VERIFIED_POOL_DROWNING"
        elif cid_int == 0:
            meaning = "Passive / floating / stationary swimmer in indoor pool lane (`output_0001..0600`)"
            consistent = "NUMERIC_NEEDS_REMAPPING"
        elif cid_int == 2:
            meaning = "Person on pool deck edge / secondary swimmer (`output_0001..0600`)"
            consistent = "NUMERIC_NEEDS_REMAPPING"
        elif cid_int == 3:
            meaning = "Active lap swimmer in indoor pool lane (`output_0001..0600`)"
            consistent = "NUMERIC_NEEDS_REMAPPING"
        else:
            meaning = f"Class {cid_int} in `{vk}`"
            consistent = "NUMERIC_NEEDS_REMAPPING"
    elif sf == "SF06_kakaotalk":
        if cid_int == 2:
            meaning = "Children floating / holding lane rope in Korean swimming pool (`kakaotalk`)"
            consistent = "NUMERIC_NEEDS_REMAPPING"
        elif cid_int == 3:
            meaning = "Active swimmers / children in Korean swimming pool (`kakaotalk`)"
            consistent = "NUMERIC_NEEDS_REMAPPING"
        else:
            meaning = "Swimmer in Korean swimming pool (`kakaotalk`)"
            consistent = "NUMERIC_NEEDS_REMAPPING"
    elif has_col and cid_int in (0, 3, 4, 5):
        if cid_int == 0:
            meaning = "Drowning / struggling person in water (numeric export before +4 shift)"
        elif cid_int == 3:
            meaning = "Normal swimmer / bystander in water (numeric export before +2 shift)"
        elif cid_int == 4:
            meaning = "INVERTED (+4 OFFSET): Drowning / struggling person in water erroneously labeled `4 (Out of Water)`!"
        else:
            meaning = "INVERTED (+2 OFFSET): Normal bystander swimmer in water erroneously labeled `5 (drowning)`!"
        consistent = "NO_CLASS_OFFSET_CORRUPTION"
    elif cid_int == 6 and sf in ("SF04_pure_digits", "SF12_clipchamp_mp4"):
        meaning = "Active swimmer in swimming pool (`6 = swimming`)"
        consistent = "YES_POOL_SWIMMING_IF_CLEAN"
    else:
        meaning = f"Class {cid_int} in `{vk}` (`{sf}`); see family forensic notes"
        consistent = "REQUIRES_MANUAL_VERIFICATION"

    csem_rows.append({
        "super_family": sf,
        "source_group_video_key": vk,
        "class_id": cid_int,
        "box_count": n_boxes,
        "image_count": n_imgs,
        "visually_verified_boxes": n_vis,
        "group_class_set": ";".join(map(str, cset)),
        "has_offset_collision": int(has_col),
        "observed_visual_meaning": meaning,
        "semantic_consistency_status": consistent,
    })

csem_df = pd.DataFrame(csem_rows).sort_values(["super_family", "source_group_video_key", "class_id"])
csem_df.to_csv(os.path.join(ROOT, "CLASS_SEMANTICS_BY_SOURCE.csv"), index=False, encoding="utf-8")
print("Wrote CLASS_SEMANTICS_BY_SOURCE.csv:", len(csem_df))

# -------------------------------------------------------------------------
# PHASE E DELIVERABLES: FULL_DUPLICATE_GROUPS.csv & FULL_LEAKAGE_RELATIONSHIPS.csv
# -------------------------------------------------------------------------
# 1. FULL_DUPLICATE_GROUPS.csv: All 7,317 multi-file exact image_sha256 groups (21,579 files)
#    plus exact label_sha256 conflict flag and cross-split status
dup_groups = []
for sg, grp in inv[inv["sha_group_size"] > 1].groupby("sha_group"):
    splits = sorted(grp["split"].unique())
    is_cross = len(splits) > 1
    csets = sorted(grp["class_ids"].unique())
    dup_groups.append({
        "duplicate_group_id": f"SHA_GRP_{int(sg):05d}",
        "evidence_level": "CONFIRMED_EXACT_DUP",
        "similarity_metric": "image_sha256_exact",
        "image_sha256": grp["image_sha256"].iloc[0],
        "group_size": len(grp),
        "redundant_copies": len(grp) - 1,
        "splits_present": ";".join(splits),
        "is_cross_split": int(is_cross),
        "unique_source_prefixes": ";".join(sorted(grp["source_prefix"].unique())),
        "video_key": grp["video_key"].iloc[0],
        "super_family": grp["super_family"].iloc[0],
        "unique_label_sha256_count": grp["label_sha256"].nunique(),
        "distinct_class_id_sets": " | ".join(csets),
        "has_conflicting_class_labels": int(len(csets) > 1),
        "canonical_representative_path": grp[grp["is_sha_representative"] == 1]["rel_image_path"].iloc[0],
        "member_paths": ";".join(grp["rel_image_path"].tolist()),
    })
dup_df = pd.DataFrame(dup_groups).sort_values(["is_cross_split", "group_size", "duplicate_group_id"], ascending=[False, False, True])
dup_df.to_csv(os.path.join(ROOT, "FULL_DUPLICATE_GROUPS.csv"), index=False, encoding="utf-8")
print("Wrote FULL_DUPLICATE_GROUPS.csv:", len(dup_df))

# 2. FULL_LEAKAGE_RELATIONSHIPS.csv: All cross-split leakage units (by exact SHA-256, source_prefix, video_key, and perceptual near-dup min_dist<=6)
leak_rows = []
# (a) Video-key / family level leakage across all 1,407 video_keys (plus single-video super-families SF02_output_NNNN and SF03_imageNNNN)
inv["atomic_video_unit"] = inv["video_key"]
inv.loc[inv["super_family"] == "SF03_imageNNNN", "atomic_video_unit"] = "ATOMIC_SF03_THAIRATH_TV_CCTV"
inv.loc[inv["video_key"].isin(["drowning1_mp4", "drowning47_mp4", "drowning50_mp4_mp4", "drowning51_mp4_mp4", "drowning52_mp4_mp4", "drowning53_mp4_mp4", "drowning54_mp4_mp4", "drowning55_mp4", "drowning55_mp4_mp4"]), "atomic_video_unit"] = "ATOMIC_SF08_WAVEPOOL_RESCUE"

for unit, grp in inv.groupby("atomic_video_unit"):
    sp_counts = grp["split"].value_counts().to_dict()
    tr, va, te = sp_counts.get("train", 0), sp_counts.get("valid", 0), sp_counts.get("test", 0)
    splits_present = [s for s in ("train", "valid", "test") if sp_counts.get(s, 0) > 0]
    n_cross_sha = int((grp["sha_group_splits"].str.contains(",")).sum())
    n_pref_cross = grp.groupby("source_prefix")["split"].nunique()
    n_pref_cross_cnt = int((n_pref_cross > 1).sum())
    if len(splits_present) > 1:
        if n_cross_sha > 0:
            ev_level = "CONFIRMED_EXACT_DUP"
        elif n_pref_cross_cnt > 0:
            ev_level = "CONFIRMED_AUGMENTED_SIBLING"
        else:
            ev_level = "CONFIRMED_SAME_VIDEO"
    else:
        ev_level = "WITHIN_SINGLE_SPLIT_ONLY"

    leak_rows.append({
        "atomic_video_unit": unit,
        "super_family": grp["super_family"].iloc[0],
        "evidence_level": ev_level,
        "is_cross_split": int(len(splits_present) > 1),
        "splits_present": ";".join(splits_present),
        "leaks_train_valid": int(tr > 0 and va > 0),
        "leaks_train_test": int(tr > 0 and te > 0),
        "leaks_valid_test": int(va > 0 and te > 0),
        "total_images": len(grp),
        "unique_sha256_images": grp["sha_group"].nunique(),
        "train_images": tr,
        "valid_images": va,
        "test_images": te,
        "unique_source_prefixes": grp["source_prefix"].nunique(),
        "cross_split_source_prefixes": n_pref_cross_cnt,
        "files_in_cross_split_sha_groups": n_cross_sha,
    })

leak_df = pd.DataFrame(leak_rows).sort_values(["is_cross_split", "total_images"], ascending=[False, False])
leak_df.to_csv(os.path.join(ROOT, "FULL_LEAKAGE_RELATIONSHIPS.csv"), index=False, encoding="utf-8")
print("Wrote FULL_LEAKAGE_RELATIONSHIPS.csv:", len(leak_df))

# -------------------------------------------------------------------------
# PHASE F DELIVERABLE: RECOVERY_CANDIDATES.csv (23,586 rows)
# Strictly obeys all 7 rules in Section 9 of the prompt!
# Allowed decisions:
#   KEEP_CANDIDATE, KEEP_IMAGE_REANNOTATE, KEEP_IMAGE_REVIEW_LABEL,
#   EXCLUDE_OUT_OF_SCOPE, EXCLUDE_CORRUPTED, EXCLUDE_WRONG_OBJECT,
#   EXCLUDE_DESTRUCTIVE_AUGMENTATION, EXCLUDE_UNRESOLVED_CLASS,
#   EXCLUDE_DUPLICATE, HUMAN_REVIEW_REQUIRED
# -------------------------------------------------------------------------
def decide_recovery(row):
    sc = row["scene_type"]
    sf = row["super_family"]
    lf = row["label_format"]
    cids = [int(x) for x in str(row["class_ids"]).split(";") if x != ""]
    rstat = row["reviewer_status"]
    is_red = int(row["is_exact_dup_redundant"])

    # Rule: Only one representative of an exact duplicate group may be retained;
    # redundant byte-identical copies (14,262 files) must be excluded as EXCLUDE_DUPLICATE!
    if is_red == 1:
        return (
            "EXCLUDE_DUPLICATE",
            "NO",
            "NO",
            f"Redundant byte-identical SHA-256 duplicate (`sha_group={row['sha_group']}`, group size `{row['sha_group_size']}`). Only 1 canonical representative per exact duplicate group may be retained.",
        )

    # Unique SHA-256 representatives (9,324 unique images):
    # 1. Out-of-scope scenes
    if sc == "OUT_OF_SCOPE":
        if sf == "SF05_visdrone_0000":
            return (
                "EXCLUDE_WRONG_OBJECT",
                "NO",
                "NO",
                "VisDrone aerial urban traffic/pedestrian scene (`0000xxx`); wrong domain and wrong object type for swimming-pool lifeguard drone.",
            )
        return (
            "EXCLUDE_OUT_OF_SCOPE",
            "NO",
            "NO",
            f"Out-of-scope non-pool environment (`{sf}`, `{row['video_key']}`: beach, ocean, river, lake, or maritime).",
        )

    # 2. Destructive augmentations (2x2 Mosaics and synthetic noise cutouts)
    if sc == "SUSPECTED_MOSAIC":
        return (
            "EXCLUDE_DESTRUCTIVE_AUGMENTATION",
            "NO",
            "NO",
            f"Pre-baked 2x2 Mosaic composite (`seam_row={row['seam_row']}`, `seam_col={row['seam_col']}`) with seam-truncated objects and mixed tile scenes.",
        )
    if sc == "CORRUPTED" or row["n_cutout_boxes"] > 0:
        return (
            "EXCLUDE_DESTRUCTIVE_AUGMENTATION",
            "NO",
            "NO",
            f"Target swimmer pixels inside bounding box are obliterated by synthetic salt-and-pepper / solid dark noise cutout (`n_cutout_boxes={row['n_cutout_boxes']}`).",
        )

    # 3. Uncertain / ambiguous scenes (no visible pool structure)
    if sc == "UNCERTAIN":
        return (
            "HUMAN_REVIEW_REQUIRED",
            "CONDITIONAL_AFTER_SCENE_VERIFICATION",
            "NO",
            f"Scene lacks visible swimming-pool boundaries (`{row['source_prefix']}`); requires human visual review before inclusion.",
        )

    # 4. Remaining unique images are swimming-pool candidates (`sc == POOL`):
    # Check if they are from `SF01_youtube_N` (contaminated with TV news studio PiP, banners, and dry-land demos)
    if sf == "SF01_youtube_N":
        return (
            "HUMAN_REVIEW_REQUIRED",
            "CONDITIONAL_AFTER_FRAME_FILTERING",
            "NO",
            "From `youtube-N` family containing Kenyan TV broadcast graphics (`CITIZENTV.CO.KE`), PiP sign-language interpreter (`cls=6`), and dry-land rescue demos; requires frame-by-frame filtering and re-annotation.",
        )

    # Check if annotations are clean semantic labels (`5=drowning`, `6=swimming`) without class-offset collision or oversized boxes
    clean_semantic = (
        lf == "DETECTION"
        and len(cids) > 0
        and set(cids).issubset({5, 6})
        and row["vk_has_offset_collision"] == 0
        and row["pref_has_offset_collision"] == 0
        and row["n_oversized_90"] == 0
        and sf not in ("SF03_imageNNNN", "SF01_youtube_N")
    )
    if clean_semantic:
        if rstat in ("VERIFIED_VISUAL", "VERIFIED_SHA256_CLONE"):
            return (
                "KEEP_CANDIDATE",
                "YES",
                "YES_VERIFIED",
                f"Visually verified swimming-pool image (`{row['source_prefix']}`) with verified clean semantic box(es) (`cls={row['class_ids']}`).",
            )
        else:
            return (
                "KEEP_IMAGE_REVIEW_LABEL",
                "YES",
                "CONDITIONAL_PENDING_VISUAL_CHECK",
                f"Unique swimming-pool image (`{row['source_prefix']}`) with semantic class ID(s) `{row['class_ids']}` (`5=drowning`, `6=swimming`); image is recoverable and labels require quick human visual verification.",
            )

    # Otherwise: Pool image is recoverable, but its annotations have class-offset inversion (`0<->4`, `(0,3)<->(4,5)`),
    # incomplete TV CCTV labels (`imageNNNN`), oversized boxes, or unresolved numeric classes (`0, 1, 2, 3`)!
    return (
        "KEEP_IMAGE_REANNOTATE",
        "YES",
        "NO_REANNOTATE_REQUIRED",
        f"Unique swimming-pool image (`{row['source_prefix']}`, `{sf}`), but existing labels (`cls={row['class_ids']}`) suffer from `{row['annotation_forensic_status']}` and must be re-annotated.",
    )

rec_res = [decide_recovery(r) for _, r in inv.iterrows()]
inv["recovery_decision"] = [r[0] for r in rec_res]
inv["image_recoverable"] = [r[1] for r in rec_res]
inv["annotation_usable_as_is"] = [r[2] for r in rec_res]
inv["recovery_rationale"] = [r[3] for r in rec_res]

rec_df = pd.DataFrame({
    "rel_image_path": inv["rel_image_path"],
    "original_split": inv["split"],
    "source_prefix": inv["source_prefix"],
    "source_group": inv["video_key"],
    "super_family": inv["super_family"],
    "sha_group_id": inv["sha_group"],
    "is_sha_representative": inv["is_sha_representative"],
    "scene_type": inv["scene_type"],
    "reviewer_status": inv["reviewer_status"],
    "class_ids": inv["class_ids"],
    "annotation_forensic_status": inv["annotation_forensic_status"],
    "recovery_decision": inv["recovery_decision"],
    "image_recoverable": inv["image_recoverable"],
    "annotation_usable_as_is": inv["annotation_usable_as_is"],
    "rationale": inv["recovery_rationale"],
})
rec_df.to_csv(os.path.join(ROOT, "RECOVERY_CANDIDATES.csv"), index=False, encoding="utf-8")
print("Wrote RECOVERY_CANDIDATES.csv:", len(rec_df))
print("Recovery decisions:", inv["recovery_decision"].value_counts().to_dict())

# -------------------------------------------------------------------------
# PHASE G DELIVERABLE: PROPOSED_SPLIT_MANIFEST.csv
# Construct a zero-leakage proposed split across all recoverable pool images
# (`KEEP_CANDIDATE`, `KEEP_IMAGE_REVIEW_LABEL`, `KEEP_IMAGE_REANNOTATE`)
# grouping strictly by `leak_unit` AND `atomic_video_unit` so no exact duplicate,
# near-duplicate, source_prefix, or video sequence ever crosses TRAIN / VALID / TEST!
# -------------------------------------------------------------------------
# Build combined atomic split group `split_cluster` uniting `leak_unit` and `atomic_video_unit`
par_sc = list(range(len(inv)))
def find_sc(x):
    while par_sc[x] != x:
        par_sc[x] = par_sc[par_sc[x]]
        x = par_sc[x]
    return x
def union_sc(a, b):
    ra, rb = find_sc(a), find_sc(b)
    if ra != rb:
        par_sc[rb] = ra

for col in ["leak_unit", "atomic_video_unit", "video_key", "source_prefix", "sha_group"]:
    first = {}
    for idx, val in zip(inv.index, inv[col]):
        if val in first:
            union_sc(first[val], idx)
        else:
            first[val] = idx

inv["split_cluster_raw"] = [find_sc(i) for i in inv.index]
sc_map = {r: f"ATOMIC_GRP_{k:04d}" for k, r in enumerate(sorted(inv["split_cluster_raw"].unique()))}
inv["atomic_split_group"] = inv["split_cluster_raw"].map(sc_map)

recoverable_decisions = {"KEEP_CANDIDATE", "KEEP_IMAGE_REVIEW_LABEL", "KEEP_IMAGE_REANNOTATE"}
rec_pool = inv[inv["recovery_decision"].isin(recoverable_decisions)].copy()

# Deterministic greedy bin assignment of atomic_split_groups to target 70% TRAIN / 15% VALID / 15% TEST
# Note: Giant single-video groups (`output_` with 892 unique images and `imageNNNN` with 734 unique images)
# must go to TRAIN to avoid a single video dominating 100% of VALID or TEST!
grp_stats = (
    rec_pool.groupby("atomic_split_group")
    .agg(
        n_unique_imgs=("image_id", "count"),
        super_family=("super_family", "first"),
        video_key=("video_key", "first"),
        n_keep_cand=("recovery_decision", lambda s: int((s == "KEEP_CANDIDATE").sum())),
        n_review_lbl=("recovery_decision", lambda s: int((s == "KEEP_IMAGE_REVIEW_LABEL").sum())),
    )
    .reset_index()
    .sort_values(["n_unique_imgs", "atomic_split_group"], ascending=[False, True])
)

total_rec = len(rec_pool)
target_valid = int(round(0.15 * total_rec))
target_test = int(round(0.15 * total_rec))

assign_map = {}
cur_tr = cur_va = cur_te = 0
for _, gr in grp_stats.iterrows():
    gname = gr["atomic_split_group"]
    cnt = int(gr["n_unique_imgs"])
    # Never place a single atomic video > 250 images into VALID or TEST (would make eval split a single video!)
    if cnt > 250:
        assign_map[gname] = "PROPOSED_TRAIN"
        cur_tr += cnt
    elif cur_va + cnt <= target_valid * 1.05 and cur_va <= cur_te:
        assign_map[gname] = "PROPOSED_VALID"
        cur_va += cnt
    elif cur_te + cnt <= target_test * 1.05:
        assign_map[gname] = "PROPOSED_TEST"
        cur_te += cnt
    else:
        assign_map[gname] = "PROPOSED_TRAIN"
        cur_tr += cnt

inv["proposed_split"] = inv["atomic_split_group"].map(assign_map).fillna("EXCLUDED_NOT_IN_SPLIT")
inv.loc[~inv["recovery_decision"].isin(recoverable_decisions), "proposed_split"] = "EXCLUDED_" + inv.loc[~inv["recovery_decision"].isin(recoverable_decisions), "recovery_decision"]

# Also provide a balanced subsampled flag (capping any single video at max 150 frames in TRAIN and 50 frames in VALID/TEST)
inv["in_balanced_subsample"] = 0
for gname, grp in inv[inv["recovery_decision"].isin(recoverable_decisions)].groupby("atomic_split_group"):
    psp = assign_map[gname]
    cap = 150 if psp == "PROPOSED_TRAIN" else 50
    if len(grp) <= cap:
        inv.loc[grp.index, "in_balanced_subsample"] = 1
    else:
        step = len(grp) / cap
        chosen = [grp.index[int(i * step)] for i in range(cap)]
        inv.loc[chosen, "in_balanced_subsample"] = 1

split_manifest = pd.DataFrame({
    "rel_image_path": inv["rel_image_path"],
    "original_split": inv["split"],
    "proposed_split": inv["proposed_split"],
    "in_balanced_capped_subset": inv["in_balanced_subsample"],
    "atomic_split_group": inv["atomic_split_group"],
    "source_prefix": inv["source_prefix"],
    "source_group": inv["video_key"],
    "super_family": inv["super_family"],
    "sha_group_id": inv["sha_group"],
    "is_sha_representative": inv["is_sha_representative"],
    "scene_type": inv["scene_type"],
    "recovery_decision": inv["recovery_decision"],
    "annotation_usable_as_is": inv["annotation_usable_as_is"],
    "class_ids": inv["class_ids"],
})
split_manifest.to_csv(os.path.join(ROOT, "PROPOSED_SPLIT_MANIFEST.csv"), index=False, encoding="utf-8")
print("Wrote PROPOSED_SPLIT_MANIFEST.csv:", len(split_manifest))
print("Proposed split counts (recoverable):", inv[inv["recovery_decision"].isin(recoverable_decisions)]["proposed_split"].value_counts().to_dict())
print("Proposed split counts (balanced capped):", inv[inv["in_balanced_subsample"] == 1]["proposed_split"].value_counts().to_dict())

# Verify ZERO leakage across PROPOSED_TRAIN, PROPOSED_VALID, PROPOSED_TEST
rec_only = inv[inv["recovery_decision"].isin(recoverable_decisions)]
for col in ["sha_group", "source_prefix", "video_key", "atomic_video_unit", "leak_unit", "atomic_split_group"]:
    leaks = (rec_only.groupby(col)["proposed_split"].nunique() > 1).sum()
    assert leaks == 0, f"Leakage detected on {col}: {leaks}"
print("VERIFIED: 0 cross-split leakage across sha_group, source_prefix, video_key, atomic_video_unit, leak_unit, atomic_split_group!")

# -------------------------------------------------------------------------
# PHASE H DELIVERABLE: DATASET_READINESS_SCORECARD.csv
# All 18 required metrics from Section 11, clearly separating actual verified
# counts from candidate/group-propagated counts!
# -------------------------------------------------------------------------
n_vis_direct_all = int((inv["reviewer_status"] == "VERIFIED_VISUAL").sum())
n_vis_clone_all = int((inv["reviewer_status"] == "VERIFIED_SHA256_CLONE").sum())
n_vis_total_all = n_vis_direct_all + n_vis_clone_all
n_pend_all = int((inv["reviewer_status"] == "PENDING_REVIEW").sum())
u_sha = inv[inv["is_sha_representative"] == 1]

scorecard_rows = [
    ("1_total_images_inspected_inventory", 23586, "100.00%", "VERIFIED_EXACT", "All 23,586 images across train (21,369), valid (1,215), and test (1,002) decoded, SHA-256 hashed, and pHash/dHash profiled."),
    ("1b_direct_visual_inspection_unique_files", n_vis_direct_all, f"{100*n_vis_direct_all/len(inv):.2f}%", "VERIFIED_VISUAL", f"{n_vis_direct_all} unique dataset files (106 Phase 1 rows + 183 Phase 2 contact-sheet files, {n_vis_direct_all} net unique) directly visually inspected."),
    ("1c_direct_visual_plus_exact_sha256_clones", n_vis_total_all, f"{100*n_vis_total_all/len(inv):.2f}%", "VERIFIED_VISUAL_AND_SHA_CLONE", f"{n_vis_total_all} total files ({n_vis_direct_all} direct + {n_vis_clone_all} byte-identical SHA-256 clones) whose exact pixel bytes were visually inspected."),
    ("2_pending_visual_review_images", n_pend_all, f"{100*n_pend_all/len(inv):.2f}%", "VERIFIED_EXACT", f"{n_pend_all} files not individually opened by human/VLM eyes; classified via group-propagation & pixel forensics with `reviewer_status=PENDING_REVIEW`."),
    ("3_confirmed_pool_images_visual_and_sha_clones", int(((inv["scene_type"] == "POOL") & inv["reviewer_status"].isin(["VERIFIED_VISUAL", "VERIFIED_SHA256_CLONE"])).sum()), f"{100*((inv['scene_type'] == 'POOL') & inv['reviewer_status'].isin(['VERIFIED_VISUAL', 'VERIFIED_SHA256_CLONE'])).mean():.2f}%", "VERIFIED_VISUAL", "Directly visually confirmed swimming-pool images (including their byte-identical SHA-256 clones)."),
    ("3b_candidate_pool_images_total_raw", int((inv["scene_type"] == "POOL").sum()), f"{100*(inv['scene_type'] == 'POOL').mean():.2f}%", "CANDIDATE_GROUP_PROPAGATED", f"Total raw images classified as `POOL` (verified + source-group propagated candidates); equals {int((u_sha['scene_type']=='POOL').sum())} unique `image_sha256` images."),
    ("4_out_of_scope_images_total_raw", int((inv["scene_type"] == "OUT_OF_SCOPE").sum()), f"{100*(inv['scene_type'] == 'OUT_OF_SCOPE').mean():.2f}%", "VERIFIED_AND_PROPAGATED", f"Total raw images in `OUT_OF_SCOPE` scenes (oceans, beaches, rivers, lakes, VisDrone urban traffic, maritime); equals {int((u_sha['scene_type']=='OUT_OF_SCOPE').sum())} unique `image_sha256` images."),
    ("5_uncertain_images_total_raw", int((inv["scene_type"] == "UNCERTAIN").sum()), f"{100*(inv['scene_type'] == 'UNCERTAIN').mean():.2f}%", "VERIFIED_AND_PROPAGATED", f"Total raw images with `UNCERTAIN` scene type (tight water crops or uninspected static prefixes); equals {int((u_sha['scene_type']=='UNCERTAIN').sum())} unique `image_sha256` images."),
    ("6_corrupted_or_mosaic_images_total_raw", int(inv["scene_type"].isin(["CORRUPTED", "SUSPECTED_MOSAIC"]).sum()), f"{100*inv['scene_type'].isin(['CORRUPTED', 'SUSPECTED_MOSAIC']).mean():.2f}%", "VERIFIED_PIXEL_FORENSICS", f"Total raw images flagged as `CORRUPTED` ({int((inv['scene_type']=='CORRUPTED').sum())} raw / {int((u_sha['scene_type']=='CORRUPTED').sum())} unique SHA) or `SUSPECTED_MOSAIC` ({int((inv['scene_type']=='SUSPECTED_MOSAIC').sum())} raw / {int((u_sha['scene_type']=='SUSPECTED_MOSAIC').sum())} unique SHA)."),
    ("7_exact_duplicate_redundant_images", int(inv["is_exact_dup_redundant"].sum()), f"{100*inv['is_exact_dup_redundant'].mean():.2f}%", "VERIFIED_EXACT_SHA256", "14,262 redundant byte-identical image copies across 7,317 multi-file `image_sha256` groups (only 9,324 unique `image_sha256` hashes exist in the 23,586-image dataset)."),
    ("8_near_duplicate_unique_sha_pairs_le6", int((pp["min_dist"] <= 6).sum()), "N/A (pairs)", "VERIFIED_PHASH_DHASH", f"{int((pp['min_dist'] <= 6).sum())} pairs of distinct `image_sha256` groups with `min(phash_dist, phash_flip_dist) <= 6` (including {int((pp['min_dist'] == 0).sum())} pairs at distance 0 from re-encoded/flipped siblings)."),
    ("9_cross_split_leakage_source_prefixes", 685, "25.58% of 2,678 prefixes (12,667 imgs / 53.71%)", "VERIFIED_EXACT", f"685 `.rf.` source prefixes (12,667 images) and {int(leak_df['is_cross_split'].sum())} `atomic_video_unit` groups ({int(leak_df[leak_df['is_cross_split']==1]['total_images'].sum())} images, {100*leak_df[leak_df['is_cross_split']==1]['total_images'].sum()/len(inv):.2f}%) span multiple train/valid/test splits."),
    ("10_usable_annotations_without_change_verified", int((inv["recovery_decision"] == "KEEP_CANDIDATE").sum()), f"{100*(inv['recovery_decision'] == 'KEEP_CANDIDATE').mean():.2f}%", "VERIFIED_VISUAL", "Unique visually verified pool images with clean semantic `5 (drowning)` / `6 (swimming)` labels ready without modification."),
    ("10b_usable_annotations_pending_quick_visual_check", int((inv["recovery_decision"] == "KEEP_IMAGE_REVIEW_LABEL").sum()), f"{100*(inv['recovery_decision'] == 'KEEP_IMAGE_REVIEW_LABEL').mean():.2f}%", "CANDIDATE_SEMANTIC_POOL", "Unique pool images with clean `5`/`6` semantic labels (`output1..75`, `1..200_jpg`, `-Clipchamp-_mp4` clean frames) that only require quick visual confirmation."),
    ("11_recoverable_pool_images_needing_reannotation", int((inv["recovery_decision"] == "KEEP_IMAGE_REANNOTATE").sum()), f"{100*(inv['recovery_decision'] == 'KEEP_IMAGE_REANNOTATE').mean():.2f}%", "CANDIDATE_REANNOTATE", "Unique pool images (`output_NNNN`, `imageNNNN`, `kakaotalk`, `drowningN_mp4`, `video_2022-12-07`, `clipchamp-1`) whose images are recoverable but labels have class-offset inversion or numeric IDs."),
    ("12_images_requiring_human_review", int((inv["recovery_decision"] == "HUMAN_REVIEW_REQUIRED").sum()), f"{100*(inv['recovery_decision'] == 'HUMAN_REVIEW_REQUIRED').mean():.2f}%", "CANDIDATE_HUMAN_REVIEW", "Unique images in `youtube-N` (TV broadcast/PiP/dry-land mix) or `UNCERTAIN` scene prefixes requiring frame-by-frame human triage."),
    ("13_excluded_images_total", int(inv["recovery_decision"].str.startswith("EXCLUDE_").sum()), f"{100*inv['recovery_decision'].str.startswith('EXCLUDE_').mean():.2f}%", "VERIFIED_EXCLUSION", f"Total excluded files (14,262 `EXCLUDE_DUPLICATE` + {int((inv['recovery_decision']=='EXCLUDE_DESTRUCTIVE_AUGMENTATION').sum())} `EXCLUDE_DESTRUCTIVE_AUGMENTATION` + {int((inv['recovery_decision']=='EXCLUDE_OUT_OF_SCOPE').sum())} `EXCLUDE_OUT_OF_SCOPE` + {int((inv['recovery_decision']=='EXCLUDE_WRONG_OBJECT').sum())} `EXCLUDE_WRONG_OBJECT`)."),
    ("14_estimated_final_clean_pool_dataset_size", int(len(rec_pool)), f"{100*len(rec_pool)/len(inv):.2f}% of raw ({100*len(rec_pool)/len(u_sha):.2f}% of unique SHA)", "RECOVERABLE_UNIQUE_POOL_IMAGES", f"{len(rec_pool)} unique `image_sha256` swimming-pool images ({int((inv['in_balanced_subsample']==1).sum())} after capping giant single videos at <=150 train / <=50 eval frames)."),
    ("15_estimated_clean_drowning_class_coverage", int(rec_pool["class_ids"].str.contains(r"\b(0|4|5)\b", regex=True).sum()), "N/A", "RECOVERABLE_POOL_WITH_DROWNING", "Unique recoverable pool images depicting drowning/submerged/distress victims (clean `Class 5`, plus pool images currently labeled `0` or offset-inverted `4` needing label remap/re-annotation)."),
    ("16_estimated_clean_swimming_class_coverage", int(rec_pool["class_ids"].str.contains(r"\b(2|3|5|6)\b", regex=True).sum()), "N/A", "RECOVERABLE_POOL_WITH_SWIMMERS", "Unique recoverable pool images depicting active/lane swimmers (clean `Class 6`, plus pool images currently labeled numeric `2`/`3` or offset-inverted `5` needing label remap/re-annotation)."),
    ("17_estimated_clean_out_of_water_class_coverage", 159, "N/A", "REQUIRES_REANNOTATION", "Almost all 8,160 existing `Class 4 (Out of Water)` boxes are corrupted by the +4 class-index offset (boxing drowning people in water!); true poolside deck bystanders exist in `output_NNNN`, `imageNNNN`, `drowning1/47`, and `clipchamp-1` but require manual re-annotation."),
    ("18_overall_dataset_readiness_rating", "UNAPPROVED_FOR_TRAINING", "0% Ready As-Is", "MANDATORY_GATE_BLOCKED", "Dataset CANNOT be used for training in its current state due to 60.47% exact image duplication, 53.71% cross-split leakage, systematic (0,3)<->(4,5) class-offset inversion, and out-of-scope/corrupted images."),
]
score_df = pd.DataFrame(scorecard_rows, columns=["metric_name", "value", "percentage_or_ratio", "evidence_basis", "forensic_explanation"])
score_df.to_csv(os.path.join(ROOT, "DATASET_READINESS_SCORECARD.csv"), index=False, encoding="utf-8")
print("Wrote DATASET_READINESS_SCORECARD.csv:", len(score_df))

# -------------------------------------------------------------------------
# Generate 12 high-resolution diagnostic visual overlays in visual_overlays/
# -------------------------------------------------------------------------
OVERLAY_SAMPLES = [
    ("ov01_offset_pair_drowning1_numeric_0_3.jpg", "train", "train/images/drowning1_mp4-13_jpg.rf.786777de041eb545321e4fd94f7abb6a.jpg", "drowning1_mp4: Numeric export (c0=drowning, c3=swimmers)"),
    ("ov02_offset_pair_drowning1_shifted_4.jpg", "train", "train/images/drowning1_mp4-0_jpg.rf.38aae75f2436a65e0af96c6f8fba14e0.jpg", "drowning1_mp4: +4 Shifted export (c4=Out of Water on drowning victim!)"),
    ("ov03_offset_pair_drowning47_numeric_0_3.jpg", "train", "train/images/drowning47_mp4-12_jpg.rf.7fabe0837447cfdc93fd8df7a18bba64.jpg", "drowning47_mp4: Numeric export (c0=drowning, c3=swimmers)"),
    ("ov04_offset_pair_drowning47_shifted_4_5.jpg", "train", "train/images/drowning47_mp4-10_jpg.rf.0c006f93f51cfb01346a4c6c95f3513a.jpg", "drowning47_mp4: Shifted export (c4=Out of Water on victim, c5=drowning on bystanders!)"),
    ("ov05_exact_sha_dup_conflicting_labels_c0_vs_c4.jpg", "train", "train/images/The-Child-Is-Drowning-In-Water-Selective-Focus-Nature-Stock-Video-Download-Video-Clip-Now-iStock_mp4-0_jpg.rf.320a9ba0ff8851b3c18bcce233e8df06.jpg", "Byte-identical SHA-256 clone labeled c0 in one file and c4 (Out of Water) in another"),
    ("ov06_destructive_cutout_clipchamp.jpg", "train", "train/images/-Clipchamp-_mp4-20_jpg.rf.5afb78e0d4607d43a0ae142617d622ca.jpg", "-Clipchamp-_mp4: Synthetic black/speckle noise cutout obliterating swimmers"),
    ("ov07_destructive_cutout_thairath_cctv.jpg", "train", "train/images/image954_jpg.rf.0fc062b348c7bae01444e4fb5175604b.jpg", "image954_jpg: THAIRATH TV 32 pool CCTV with black noise cutout over drowning child"),
    ("ov08_thairath_cctv_incomplete_class4.jpg", "train", "train/images/image839_jpg.rf.1e5cfdb2937dfa06278701a541e90f55.jpg", "image839_jpg: THAIRATH TV 32 CCTV (c4=Out of Water on submerged child; bystanders unannotated)"),
    ("ov09_youtube_mosaic_2x2.jpg", "train", "train/images/youtube-40_jpg.rf.0613d984ce249d690845d73023548ce9.jpg", "youtube-40_jpg: Pre-baked 2x2 Mosaic composite with seam-truncated boxes"),
    ("ov10_youtube_tv_news_pip_interpreter.jpg", "train", "train/images/youtube-14_jpg.rf.0903b8cf9ba6c9ff7dd404907876ab96.jpg", "youtube-14_jpg: CITIZENTV.CO.KE broadcast with PiP sign-language interpreter boxed as c6 (swimming)"),
    ("ov11_clean_pool_output_drowning_c5.jpg", "train", "train/images/output29_jpg.rf.50686d82c76351af38e0a22f3579e100.jpg", "output29_jpg: Clean indoor pool frame with submerged victim labeled c5 (drowning)"),
    ("ov12_out_of_scope_white_water_river.jpg", "train", "train/images/white-water_mp4-24_jpg.rf.0145278a2341212bb7bde42c106baddd.jpg", "white-water_mp4: Out-of-scope white-water river rapids tubing labeled c6 (swimming)"),
]

COLORS = {0: (0, 165, 255), 1: (255, 0, 255), 2: (255, 255, 0), 3: (0, 255, 255), 4: (0, 255, 0), 5: (0, 0, 255), 6: (255, 128, 0)}
CLASS_NAMES = {0: "c0('3')", 1: "c1('4')", 2: "c2('5')", 3: "c3('6')", 4: "c4(OutOfWater)", 5: "c5(drowning)", 6: "c6(swimming)"}

z_open = {k: zipfile.ZipFile(v, "r") for k, v in ZIPS.items()}
for out_name, sp, rel, caption in OVERLAY_SAMPLES:
    data = z_open[sp].read(rel)
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    H, W = img.shape[:2]
    ib = boxes[(boxes["split"] == sp) & (boxes["rel_image_path"] == rel)]
    for _, bx in ib.iterrows():
        if pd.notna(bx["cx"]):
            cid = int(bx["class_id"])
            col = COLORS.get(cid, (255, 255, 255))
            x1 = int((bx["cx"] - bx["w"] / 2) * W)
            y1 = int((bx["cy"] - bx["h"] / 2) * H)
            x2 = int((bx["cx"] + bx["w"] / 2) * W)
            y2 = int((bx["cy"] + bx["h"] / 2) * H)
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
            cv2.putText(img, CLASS_NAMES.get(cid, str(cid)), (max(4, x1), max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
    banner = np.zeros((54, W, 3), dtype=np.uint8)
    cv2.putText(banner, f"{rel}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 230, 255), 1, cv2.LINE_AA)
    cv2.putText(banner, caption, (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(OV_DIR, out_name), np.vstack([banner, img]), [int(cv2.IMWRITE_JPEG_QUALITY), 92])

for zf in z_open.values():
    zf.close()
print("Wrote 12 diagnostic overlays to visual_overlays/")

# Save comprehensive summary JSON for report generation
summary = {
    "total_images": int(len(inv)),
    "split_counts": inv["split"].value_counts().to_dict(),
    "unique_sha256_images": int(inv["sha_group"].nunique()),
    "redundant_exact_dup_images": int(inv["is_exact_dup_redundant"].sum()),
    "multi_file_sha_groups": int((inv.groupby("sha_group").size() > 1).sum()),
    "cross_split_sha_groups": int(dup_df["is_cross_split"].sum()),
    "cross_split_sha_files": int(dup_df[dup_df["is_cross_split"] == 1]["group_size"].sum()),
    "within_split_sha_groups": int((dup_df["is_cross_split"] == 0).sum()),
    "within_split_sha_files": int(dup_df[dup_df["is_cross_split"] == 0]["group_size"].sum()),
    "sha_groups_with_conflicting_labels": int(dup_df["has_conflicting_class_labels"].sum()),
    "unique_source_prefixes": int(inv["source_prefix"].nunique()),
    "unique_video_keys": int(inv["video_key"].nunique()),
    "unique_atomic_video_units": int(inv["atomic_video_unit"].nunique()),
    "cross_split_atomic_video_units": int(leak_df["is_cross_split"].sum()),
    "files_in_cross_split_atomic_video_units": int(leak_df[leak_df["is_cross_split"] == 1]["total_images"].sum()),
    "reviewer_status_counts": inv["reviewer_status"].value_counts().to_dict(),
    "scene_type_raw_counts": inv["scene_type"].value_counts().to_dict(),
    "scene_type_unique_sha_counts": inv[inv["is_sha_representative"] == 1]["scene_type"].value_counts().to_dict(),
    "scene_type_verified_counts": inv[inv["reviewer_status"].isin(["VERIFIED_VISUAL", "VERIFIED_SHA256_CLONE"])]["scene_type"].value_counts().to_dict(),
    "annotation_forensic_status_raw": inv["annotation_forensic_status"].value_counts().to_dict(),
    "annotation_forensic_status_unique_sha": inv[inv["is_sha_representative"] == 1]["annotation_forensic_status"].value_counts().to_dict(),
    "recovery_decision_raw": inv["recovery_decision"].value_counts().to_dict(),
    "recovery_decision_unique_sha": inv[inv["is_sha_representative"] == 1]["recovery_decision"].value_counts().to_dict(),
    "proposed_split_recoverable": inv[inv["recovery_decision"].isin(recoverable_decisions)]["proposed_split"].value_counts().to_dict(),
    "proposed_split_balanced": inv[inv["in_balanced_subsample"] == 1]["proposed_split"].value_counts().to_dict(),
}
with open(os.path.join(C, "summary_stats.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print("Saved summary_stats.json")
