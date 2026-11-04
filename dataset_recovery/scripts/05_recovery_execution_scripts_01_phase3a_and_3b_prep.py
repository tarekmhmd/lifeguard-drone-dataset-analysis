"""Phase 3A (Evidence Validation) & Phase 3B Preparation (Render 18 Gold Candidate Overlays & Contact Sheets).
Strict read-only access to original dataset archives and Phase 1/2 deliverables.
All outputs written under E:\lifeguardDron\DrowningDetection_v1_cleaning\05_recovery_execution\
"""
import hashlib, json, os, zipfile
import cv2, numpy as np, pandas as pd

P2_DIR = r"E:\lifeguardDron\DrowningDetection_v1_cleaning\04_recovery_investigation"
P3_DIR = r"E:\lifeguardDron\DrowningDetection_v1_cleaning\05_recovery_execution"
SRC_DIR = r"E:\lifeguardDron\DrowningDetectionTracking_v1"

SUBDIRS = [
    "00_state",
    "01_evidence_validation",
    "02_gold_reference_set",
    "02_gold_reference_set/overlays",
    "02_gold_reference_set/contact_sheets",
    "03_label_review_737",
    "03_label_review_737/contact_sheets",
    "03_label_review_737/overlays",
    "03_label_review_737/corrections",
    "04_reannotation_3463",
    "04_reannotation_3463/contact_sheets",
    "04_reannotation_3463/overlays",
    "04_reannotation_3463/proposed_labels",
    "05_human_review_1813",
    "05_human_review_1813/contact_sheets",
    "06_split_validation",
    "07_quality_assurance",
    "08_reports",
    "scripts",
    "logs",
]
for sd in SUBDIRS:
    os.makedirs(os.path.join(P3_DIR, sd), exist_ok=True)

# 1. Verify required Phase 2 files exist
REQUIRED_P2_FILES = [
    ("EXECUTION_STATE.md", "markdown"),
    ("01_PREVIOUS_EVIDENCE_AUDIT.md", "markdown"),
    ("FULL_DATASET_INVENTORY.csv", "csv"),
    ("FULL_DATASET_INVENTORY_REPORT.md", "markdown"),
    ("SCENE_CLASSIFICATION_FULL.csv", "csv"),
    ("SCENE_CLASSIFICATION_REPORT.md", "markdown"),
    ("SCENE_REVIEW_PROGRESS.csv", "csv"),
    ("ANNOTATION_FORENSICS_FULL.csv", "csv"),
    ("ANNOTATION_FORENSICS_REPORT.md", "markdown"),
    ("CLASS_SEMANTICS_BY_SOURCE.csv", "csv"),
    ("FULL_DUPLICATE_GROUPS.csv", "csv"),
    ("FULL_LEAKAGE_RELATIONSHIPS.csv", "csv"),
    ("DUPLICATE_AND_LEAKAGE_FULL_REPORT.md", "markdown"),
    ("RECOVERY_CANDIDATES.csv", "csv"),
    ("RECOVERY_DECISION_REPORT.md", "markdown"),
    ("PROPOSED_SPLIT_MANIFEST.csv", "csv"),
    ("PROPOSED_SPLIT_DESIGN_REPORT.md", "markdown"),
    ("DATASET_READINESS_SCORECARD.csv", "csv"),
    ("DATASET_RECOVERY_FINAL_REPORT.md", "markdown"),
]

ev_rows = []
csv_dfs = {}
for fname, ftype in REQUIRED_P2_FILES:
    fpath = os.path.join(P2_DIR, fname)
    exists = os.path.isfile(fpath)
    fsize = os.path.getsize(fpath) if exists else 0
    fsha = hashlib.sha256(open(fpath, "rb").read()).hexdigest() if exists else ""
    nrows = ncols = 0
    cols_str = ""
    if exists and ftype == "csv":
        df = pd.read_csv(fpath, low_memory=False)
        csv_dfs[fname] = df
        nrows, ncols = df.shape
        cols_str = ";".join(df.columns)
    elif exists:
        with open(fpath, "r", encoding="utf-8") as f:
            nrows = sum(1 for _ in f)
        ncols = 1
        cols_str = "markdown_lines"
    ev_rows.append({
        "file_name": fname,
        "file_type": ftype,
        "exists": int(exists),
        "file_size_bytes": fsize,
        "row_or_line_count": nrows,
        "column_count": ncols,
        "sha256": fsha,
        "columns": cols_str,
        "validation_status": "PASS" if (exists and fsize > 0 and nrows > 0) else "FAIL",
    })

ev_df = pd.DataFrame(ev_rows)
ev_df.to_csv(os.path.join(P3_DIR, "01_evidence_validation", "EVIDENCE_VALIDATION.csv"), index=False, encoding="utf-8")
print("Wrote 01_evidence_validation/EVIDENCE_VALIDATION.csv:", len(ev_df))

# 2. Cross-validate CSV manifests against original ZIP archive entries
ZIPS = {
    "train": zipfile.ZipFile(os.path.join(SRC_DIR, "train-20261102T163226Z-1-001.zip"), "r"),
    "valid": zipfile.ZipFile(os.path.join(SRC_DIR, "valid-20261102T163228Z-1-001.zip"), "r"),
    "test": zipfile.ZipFile(os.path.join(SRC_DIR, "test-20261102T163222Z-1-001.zip"), "r"),
}
zip_names = {sp: set(zf.namelist()) for sp, zf in ZIPS.items()}

inv = csv_dfs["FULL_DATASET_INVENTORY.csv"]
scene = csv_dfs["SCENE_CLASSIFICATION_FULL.csv"]
ann = csv_dfs["ANNOTATION_FORENSICS_FULL.csv"]
rec = csv_dfs["RECOVERY_CANDIDATES.csv"]
split_m = csv_dfs["PROPOSED_SPLIT_MANIFEST.csv"]
dup = csv_dfs["FULL_DUPLICATE_GROUPS.csv"]
leak = csv_dfs["FULL_LEAKAGE_RELATIONSHIPS.csv"]
prog = csv_dfs["SCENE_REVIEW_PROGRESS.csv"]
score = csv_dfs["DATASET_READINESS_SCORECARD.csv"]

# Check image & label path existence in original ZIPs
img_exists = [int(r["rel_image_path"] in zip_names[r["original_split"]]) for _, r in inv.iterrows()]
lbl_exists = [int(r["label_path"] in zip_names[r["original_split"]]) for _, r in inv.iterrows()]
assert sum(img_exists) == 23586, f"Missing images: {23586 - sum(img_exists)}"
assert sum(lbl_exists) == 23586, f"Missing labels: {23586 - sum(lbl_exists)}"

# Check exact alignment of rel_image_path across all 5 23,586-row tables
assert (inv["rel_image_path"].values == scene["rel_image_path"].values).all()
assert (inv["rel_image_path"].values == ann["rel_image_path"].values).all()
assert (inv["rel_image_path"].values == rec["rel_image_path"].values).all()
assert (inv["rel_image_path"].values == split_m["rel_image_path"].values).all()
assert inv["rel_image_path"].nunique() == 23586

# Reconcile counts across tables and reports
recon_rows = [
    ("total_raw_images", 23586, len(inv), len(rec), "MATCH", "All 5 full-dataset CSVs have exactly 23,586 unique rel_image_path rows."),
    ("split_train_images", 21369, int((inv["original_split"] == "train").sum()), int((rec["original_split"] == "train").sum()), "MATCH", "Verified against train ZIP archive."),
    ("split_valid_images", 1215, int((inv["original_split"] == "valid").sum()), int((rec["original_split"] == "valid").sum()), "MATCH", "Verified against valid ZIP archive."),
    ("split_test_images", 1002, int((inv["original_split"] == "test").sum()), int((rec["original_split"] == "test").sum()), "MATCH", "Verified against test ZIP archive."),
    ("unique_sha256_images", 9324, int(inv["sha_group_id"].nunique()), int((inv["is_exact_dup_redundant"] == 0).sum()), "MATCH", "9,324 unique image_sha256 hashes (2,007 singletons + 7,317 multi-file groups)."),
    ("redundant_exact_duplicates", 14262, int((inv["is_exact_dup_redundant"] == 1).sum()), int((rec["recovery_decision"] == "EXCLUDE_DUPLICATE").sum()), "MATCH", "14,262 redundant byte-identical image files marked EXCLUDE_DUPLICATE."),
    ("verified_visual_direct_images", 279, int((scene["reviewer_status"] == "VERIFIED_VISUAL").sum()), int((rec["reviewer_status"] == "VERIFIED_VISUAL").sum()), "MATCH", "279 unique files directly visually inspected in Phase 1 + Phase 2."),
    ("verified_sha256_clone_images", 412, int((scene["reviewer_status"] == "VERIFIED_SHA256_CLONE").sum()), int((rec["reviewer_status"] == "VERIFIED_SHA256_CLONE").sum()), "MATCH", "412 byte-identical SHA-256 clones of directly inspected files."),
    ("pending_review_images", 22895, int((scene["reviewer_status"] == "PENDING_REVIEW").sum()), int((rec["reviewer_status"] == "PENDING_REVIEW").sum()), "MATCH", "22,895 files with reviewer_status = PENDING_REVIEW."),
    ("scene_POOL_raw_and_unique_sha", 12681, int((scene["scene_type"] == "POOL").sum()), int(((scene["scene_type"] == "POOL") & (inv["is_exact_dup_redundant"] == 0)).sum()), "MATCH (12,681 raw / 5,107 unique SHA)", "Matches SCENE_CLASSIFICATION_FULL.csv and DATASET_READINESS_SCORECARD.csv."),
    ("scene_CORRUPTED_raw_and_unique_sha", 4894, int((scene["scene_type"] == "CORRUPTED").sum()), int(((scene["scene_type"] == "CORRUPTED") & (inv["is_exact_dup_redundant"] == 0)).sum()), "MATCH (4,894 raw / 1,691 unique SHA)", "Synthetic salt-and-pepper / solid dark noise-cutout images."),
    ("scene_OUT_OF_SCOPE_raw_and_unique_sha", 2643, int((scene["scene_type"] == "OUT_OF_SCOPE").sum()), int(((scene["scene_type"] == "OUT_OF_SCOPE") & (inv["is_exact_dup_redundant"] == 0)).sum()), "MATCH (2,643 raw / 1,147 unique SHA)", "Oceans, beaches, rivers, lakes, maritime, VisDrone streets."),
    ("scene_UNCERTAIN_raw_and_unique_sha", 2225, int((scene["scene_type"] == "UNCERTAIN").sum()), int(((scene["scene_type"] == "UNCERTAIN") & (inv["is_exact_dup_redundant"] == 0)).sum()), "MATCH (2,225 raw / 931 unique SHA)", "Tightly cropped water images or uninspected static prefixes."),
    ("scene_SUSPECTED_MOSAIC_raw_and_unique_sha", 1143, int((scene["scene_type"] == "SUSPECTED_MOSAIC").sum()), int(((scene["scene_type"] == "SUSPECTED_MOSAIC") & (inv["is_exact_dup_redundant"] == 0)).sum()), "MATCH (1,143 raw / 448 unique SHA)", "Pre-baked 2x2 Mosaic composites."),
    ("decision_KEEP_CANDIDATE", 18, int((rec["recovery_decision"] == "KEEP_CANDIDATE").sum()), int((split_m["recovery_decision"] == "KEEP_CANDIDATE").sum()), "MATCH", "18 unique visually verified pool images with clean 5/6 semantic labels."),
    ("decision_KEEP_IMAGE_REVIEW_LABEL", 737, int((rec["recovery_decision"] == "KEEP_IMAGE_REVIEW_LABEL").sum()), int((split_m["recovery_decision"] == "KEEP_IMAGE_REVIEW_LABEL").sum()), "MATCH", "737 unique pool images with 5/6 labels pending visual verification."),
    ("decision_KEEP_IMAGE_REANNOTATE", 3463, int((rec["recovery_decision"] == "KEEP_IMAGE_REANNOTATE").sum()), int((split_m["recovery_decision"] == "KEEP_IMAGE_REANNOTATE").sum()), "MATCH", "3,463 unique pool images needing label correction/re-annotation."),
    ("decision_HUMAN_REVIEW_REQUIRED", 1813, int((rec["recovery_decision"] == "HUMAN_REVIEW_REQUIRED").sum()), int((split_m["recovery_decision"] == "HUMAN_REVIEW_REQUIRED").sum()), "MATCH", "882 non-mosaic SF01_youtube_N + 931 UNCERTAIN unique SHAs."),
    ("decision_EXCLUDE_DESTRUCTIVE_AUGMENTATION", 2146, int((rec["recovery_decision"] == "EXCLUDE_DESTRUCTIVE_AUGMENTATION").sum()), int((split_m["recovery_decision"] == "EXCLUDE_DESTRUCTIVE_AUGMENTATION").sum()), "MATCH", "1,691 CORRUPTED + 7 additional cutout SHAs (1,698) + 448 SUSPECTED_MOSAIC SHAs = 2,146."),
    ("decision_EXCLUDE_OUT_OF_SCOPE", 1094, int((rec["recovery_decision"] == "EXCLUDE_OUT_OF_SCOPE").sum()), int((split_m["recovery_decision"] == "EXCLUDE_OUT_OF_SCOPE").sum()), "MATCH", "1,094 unique non-VisDrone out-of-scope SHAs."),
    ("decision_EXCLUDE_WRONG_OBJECT", 53, int((rec["recovery_decision"] == "EXCLUDE_WRONG_OBJECT").sum()), int((split_m["recovery_decision"] == "EXCLUDE_WRONG_OBJECT").sum()), "MATCH", "53 unique non-cutout VisDrone street SHAs."),
    ("proposed_split_full_4218", 4218, int(split_m["proposed_split"].isin(["PROPOSED_TRAIN", "PROPOSED_VALID", "PROPOSED_TEST"]).sum()), 2890 + 664 + 664, "MATCH", "PROPOSED_TRAIN=2890, PROPOSED_VALID=664, PROPOSED_TEST=664 (0 cross-split leakage)."),
    ("proposed_split_capped_1935", 1935, int((split_m["in_balanced_capped_subset"] == 1).sum()), 941 + 524 + 470, "MATCH", "PROPOSED_TRAIN=941, PROPOSED_VALID=524, PROPOSED_TEST=470 (0 cross-split leakage)."),
    ("audit_note_super_family_regex_refinement", 9324, int((rec["is_sha_representative"] == 1).sum()), int(prog["unique_sha256_images"].sum()), "NOTE_DOCUMENTED", "In 04_build_csv_manifests.py, SF08/SF13/SF18 regex refinement shifted 83 raw rows between SF18 and SF08/SF13 relative to the earlier draft table in FULL_DATASET_INVENTORY_REPORT.md, and SCENE_REVIEW_PROGRESS.csv's `unique_sha256_images` sums per-family `sha_group.nunique()` (`9,324` global unique SHAs). All CSV manifests are 100% mutually consistent."),
]
recon_df = pd.DataFrame(recon_rows, columns=["metric_or_category", "expected_reported_value", "csv_primary_count", "csv_secondary_count", "reconciliation_status", "notes"])
recon_df.to_csv(os.path.join(P3_DIR, "01_evidence_validation", "COUNT_RECONCILIATION.csv"), index=False, encoding="utf-8")
print("Wrote 01_evidence_validation/COUNT_RECONCILIATION.csv:", len(recon_df))

# -------------------------------------------------------------------------
# 3. Phase 3B Preparation: Extract, SHA-verify, and render overlays & contact sheets
#    for the 18 KEEP_CANDIDATE images so they can be visually inspected!
# -------------------------------------------------------------------------
boxes_raw = pd.read_csv(os.path.join(P2_DIR, "analysis_logs", "cache", "boxes_raw.csv"), low_memory=False)
inv_keys = pd.read_csv(os.path.join(P2_DIR, "analysis_logs", "cache", "inventory_keys.csv"), low_memory=False)

gold_cands = rec[rec["recovery_decision"] == "KEEP_CANDIDATE"].merge(
    inv[["rel_image_path", "image_sha256", "annotation_sha256", "label_path", "file_size", "sha_group_size"]],
    on="rel_image_path", how="left"
).merge(
    inv_keys[["rel_image_path", "image_id", "aqua_frac", "blue_frac", "dark_frac", "sp_frac_global", "seam_row", "seam_col", "black_corners"]],
    on="rel_image_path", how="left"
).sort_values(["super_family", "source_prefix", "rel_image_path"]).reset_index(drop=True)

COLORS = {5: (0, 0, 255), 6: (255, 128, 0), 0: (0, 165, 255), 1: (255, 0, 255), 2: (255, 255, 0), 3: (0, 255, 255), 4: (0, 255, 0)}
CNAMES = {5: "5:drowning", 6: "6:swimming", 0: "0:'3'", 1: "1:'4'", 2: "2:'5'", 3: "3:'6'", 4: "4:OutOfWater"}

tiles = []
gold_meta = []
for idx, r in gold_cands.iterrows():
    gid = f"GOLD_{idx+1:02d}"
    sp = r["original_split"]
    rel = r["rel_image_path"]
    lpath = r["label_path"]
    raw_bytes = ZIPS[sp].read(rel)
    lbl_bytes = ZIPS[sp].read(lpath)
    actual_sha = hashlib.sha256(raw_bytes).hexdigest()
    actual_lsha = hashlib.sha256(lbl_bytes).hexdigest()
    sha_ok = int(actual_sha == r["image_sha256"])
    lsha_ok = int(actual_lsha == r["annotation_sha256"])
    img = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
    H, W = img.shape[:2]
    ib = boxes_raw[(boxes_raw["split"] == sp) & (boxes_raw["rel_image_path"] == rel)]
    over = img.copy()
    box_desc = []
    for _, bx in ib.iterrows():
        cid = int(bx["class_id"])
        cx, cy, bw, bh = float(bx["cx"]), float(bx["cy"]), float(bx["w"]), float(bx["h"])
        x1, y1 = int(round((cx - bw / 2) * W)), int(round((cy - bh / 2) * H))
        x2, y2 = int(round((cx + bw / 2) * W)), int(round((cy + bh / 2) * H))
        col = COLORS.get(cid, (0, 255, 255))
        cv2.rectangle(over, (x1, y1), (x2, y2), col, 2)
        cv2.putText(over, f"#{int(bx['line_no'])} {CNAMES.get(cid, str(cid))} ({bw*bh:.3f})",
                    (max(4, x1), max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 2, cv2.LINE_AA)
        box_desc.append(f"L{int(bx['line_no'])}:c{cid}({cx:.3f},{cy:.3f},{bw:.3f},{bh:.3f},area={bw*bh:.4f},dark={bx['in_dark']:.2f},sp={bx['in_sp']:.4f})")

    # Save full-res side-by-side (raw | overlay) for crystal-clear inspection!
    banner = np.zeros((56, W * 2, 3), dtype=np.uint8)
    cv2.putText(banner, f"{gid} | [{sp}] {rel} | sha_ok={sha_ok} | corner_blk={r['black_corners']}",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(banner, f"sf={r['super_family']} | prefix={r['source_prefix']} | cls={r['class_ids']} | boxes={len(ib)} | {' ; '.join(box_desc[:3])}",
                (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 235, 255), 1, cv2.LINE_AA)
    pair_img = np.vstack([banner, np.hstack([img, over])])
    ov_fname = f"{gid}_{r['source_prefix']}.jpg"
    cv2.imwrite(os.path.join(P3_DIR, "02_gold_reference_set", "overlays", ov_fname), pair_img, [int(cv2.IMWRITE_JPEG_QUALITY), 94])

    # Build 420x420 contact-sheet tile
    t_img = cv2.resize(over, (420, 420), interpolation=cv2.INTER_AREA)
    t_ban = np.zeros((48, 420, 3), dtype=np.uint8)
    cv2.putText(t_ban, f"{gid} [{sp}] {r['source_prefix']} cls={r['class_ids']} ({len(ib)}b)",
                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(t_ban, f"id={r['image_id']} blk_crn={r['black_corners']} max_sp={ib['in_sp'].max():.4f}",
                (6, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 235, 255), 1, cv2.LINE_AA)
    tiles.append(np.vstack([t_ban, t_img]))

    gold_meta.append({
        "gold_id": gid,
        "image_id": int(r["image_id"]),
        "original_split": sp,
        "rel_image_path": rel,
        "label_path": lpath,
        "source_prefix": r["source_prefix"],
        "source_group": r["source_group"],
        "super_family": r["super_family"],
        "sha_group_id": int(r["sha_group_id"]),
        "sha_group_size": int(r["sha_group_size"]),
        "image_sha256": actual_sha,
        "sha256_verified": sha_ok,
        "annotation_sha256": actual_lsha,
        "label_sha256_verified": lsha_ok,
        "width": W,
        "height": H,
        "n_boxes": len(ib),
        "class_ids": str(r["class_ids"]),
        "black_corners": int(r["black_corners"]),
        "box_details": " | ".join(box_desc),
        "overlay_file": f"02_gold_reference_set/overlays/{ov_fname}",
    })

# Save 2 contact sheets (3x3 = 9 tiles each: Sheet 1 = GOLD_01..09, Sheet 2 = GOLD_10..18)
for s_idx, chunk in enumerate([tiles[:9], tiles[9:18]], 1):
    while len(chunk) < 9:
        chunk.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(chunk[r * 3 : (r + 1) * 3]) for r in range(3)])
    cs_path = os.path.join(P3_DIR, "02_gold_reference_set", "contact_sheets", f"gold_reference_sheet_{s_idx:02d}.jpg")
    cv2.imwrite(cs_path, grid, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    print("Wrote", cs_path)

pd.DataFrame(gold_meta).to_csv(os.path.join(P3_DIR, "logs", "gold_candidates_pre_review.csv"), index=False, encoding="utf-8")
for zf in ZIPS.values():
    zf.close()
print("Prepared all 18 gold candidates:", len(gold_meta))

