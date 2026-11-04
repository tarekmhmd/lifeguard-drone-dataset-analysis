# Phase B — Full Dataset Inventory Report
## `DrowningDetectionTracking_v1` | Autonomous AI Lifeguard Drone (Swimming Pools Only)

**Report Date:** 2026-11-03  
**Dataset Status:** **`UNAPPROVED FOR TRAINING`**  
**Manifest Deliverable:** [`FULL_DATASET_INVENTORY.csv`](file:///E:/lifeguardDron/DrowningDetection_v1_cleaning/04_recovery_investigation/FULL_DATASET_INVENTORY.csv) (`23,586` rows × `20` columns)

---

## 1. Executive Summary of Full-Dataset Inventory

A deterministic, read-only byte-level and pixel-level pass (`01_extract_inventory.py`) was executed across all three original split archives (`train-20261102T163226Z-1-001.zip`, `valid-20261102T163228Z-1-001.zip`, `test-20261102T163222Z-1-001.zip`). Every image and label file was decoded, hashed (`SHA-256`), perceptually fingerprinted (`64-bit DCT pHash`, horizontally flipped `pHash_flip`, `64-bit dHash`), and mapped to its `.rf.` `source_prefix`, underlying `video_key`, and one of **17 exhaustive Super-Families**.

### Key Inventory Highlights
- **Total Raw Images:** **`23,586`** (`train = 21,369` [`90.60%`], `valid = 1,215` [`5.15%`], `test = 1,002` [`4.25%`]).
- **Image Integrity & Dimensions:** **`100.00%`** (`23,586 / 23,586`) are valid `640×640` RGB JPEGs (`0` corrupt/unreadable image files). File sizes range from `6,282` bytes to `176,129` bytes (mean `55.4 KB`).
- **Label File Pairing:** **`100.00%`** (`23,586 / 23,586`) have a matching `.txt` file (`0` missing label files, `0` orphan label files).
  - `23,570` (`99.93%`) are standard 5-field YOLO `DETECTION` files (`49,021` boxes).
  - `9` (`0.04%`) are `MIXED_DET_POLY` files (`19` valid 5-field boxes + `9` segmentation polygons in `3-727-...-iStock_mp4-3_jpg`).
  - `6` (`0.03%`) are `POLYGON_ONLY` files (`6` segmentation polygons in `3-727-...-iStock_mp4-2_jpg`).
  - `1` (`0.004%`) is `EMPTY` (`0` bytes: `test/labels/MVI_1627_VIS_frame20_jpg.rf.098ac3dc01adebe600fb12fc9cd1fa60.txt`).
- **True Unique Image Count (`image_sha256`):** Only **`9,324` unique `image_sha256` byte streams** exist across the `23,586` image files! The remaining **`14,262` files (`60.47%`)** are redundant byte-identical copies produced by Roboflow's `"3x versions per source image"` export and repeated sub-dataset merges.

---

## 2. Complete Breakdown by the 17 Exhaustive Super-Families

Every one of the `23,586` images (`2,678` `.rf.` `source_prefix` groups and `1,407` `video_key` groups) belongs to one of the 17 Super-Families below:

| Super-Family | Unique `video_key`s | Unique `.rf.` Prefixes | Raw Images (`% of 23,586`) | Unique `image_sha256` (`% of 9,324`) | Redundant Exact Dups | `train` / `valid` / `test` | Distinct Class IDs Present | Primary Visual Domain & Source Origin |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`SF01_youtube_N`** | 126 | 126 | **3,806** (`16.14%`) | **1,317** (`14.12%`) | 2,489 | `3,573 / 160 / 73` | `0,1,3,4,5,6` | Indoor pool videos mixed with Kenyan TV news (`CITIZENTV.CO.KE`), PiP interpreter, dry-land demos, and 2×2 Mosaics |
| **`SF08_drowningN_mp4`** | 13 | 234 | **3,593** (`15.23%`) | **1,407** (`15.09%`) | 2,186 | `3,275 / 199 / 119` | `0,3,4,5,6` | Wave-pool blue-tube rescue (`drowning1/47/50..55_mp4`) & pool close-ups (`drowning20`, `drowning2`), with `(0,3)<->(4,5)` shift |
| **`SF04_pure_digits`** | 334 | 334 | **3,121** (`13.23%`) | **1,087** (`11.66%`) | 2,034 | `2,958 / 116 / 47` | `0,1,3,4,5,6` | 4 sub-series: `1..200_jpg` (indoor pool), `1..999_png` (dark sea), `3000..12000_jpg` (aerial beach/ocean), `130000_jpg` (aerial pool) |
| **`SF18_other_static`** | 309 | 310 | **2,651** (`11.24%`) | **1,069** (`11.47%`) | 1,582 | `2,395 / 145 / 111` | `0,3,4,5,6` | Mixed web/stock photos (Dreamstime/123RF pool & ocean drowning images) with `0<->4` class-index offset |
| **`SF02_output_NNNN`** | 2 | 524 | **2,429** (`10.30%`) | **951** (`10.20%`) | 1,478 | `2,217 / 137 / 75` | `0,2,3,5` | Two overhead indoor pool videos: `output_0001..0600` (`2,276` imgs, `cls=0,2,3`) & `output1..75` (`153` imgs, `cls=5` drowning) |
| **`SF03_imageNNNN`** | 463 | 463 | **1,916** (`8.12%`) | **734** (`7.87%`) | 1,182 | `1,722 / 112 / 82` | `0,1,2,4` | Single Thai TV news broadcast (`THAIRATH TV 32`) of backyard pool CCTV (`461` prefixes have only `cls=4` on drowning child!) |
| **`SF13_stock_named`** | 26 | 95 | **1,706** (`7.23%`) | **721** (`7.73%`) | 985 | `1,353 / 113 / 240` | `0,3,4` | Named iStock/stock videos & photos (mix of pool child drowning and open-water struggles; includes the 15 polygon files) |
| **`SF10_video_2022_12_07`** | 4 | 57 | **1,258** (`5.33%`) | **512** (`5.49%`) | 746 | `1,150 / 61 / 47` | `0,3,4` | Four staged indoor tiled swimming pool clips (`23-08-07/12/15/17_mp4`) with `0<->4` class-index offset |
| **`SF17_other_video_mp4_mov`** | 6 | 101 | **803** (`3.40%`) | **367** (`3.94%`) | 436 | `607 / 115 / 81` | `0,3,4,5,6` | Out-of-scope videos: `white-water_mp4` (`354` river tubing), `10_mov` (`184` murky river), `test1-1..8_mp4` (`265` sea/open water) |
| **`SF12_clipchamp_mp4`** | 2 | 85 | **612** (`2.59%`) | **295** (`3.16%`) | 317 | `561 / 34 / 17` | `0,3,4,5,6` | `Untitled-video-Made-with-Clipchamp-1-_mp4` (`472` Helsinki pool CCTV) + `-Clipchamp-_mp4` (`140` outdoor pool w/ heavy cutouts) |
| **`SF06_kakaotalk`** | 8 | 150 | **578** (`2.45%`) | **355** (`3.81%`) | 223 | `508 / 60 / 10` | `0,2,3` | Eight Korean swimming-pool photo batches (`KakaoTalk_20230504_*` & `20230509_*`), 100% pool scenes with numeric `0,2,3` |
| **`SF15_img_pic_photo`** | 34 | 34 | **339** (`1.44%`) | **138** (`1.48%`) | 201 | `308 / 17 / 14` | `0,3,4` | Static web images (`images-1..30`, `img_*`, `pic*`): underwater pool photos, ocean hands, and `"Drowning Myth"` infographic |
| **`SF11_videoN_mp4`** | 10 | 64 | **303** (`1.28%`) | **171** (`1.83%`) | 132 | `278 / 21 / 4` | `0,4` | Open ocean surf/beach clips (`video40-1..13_mp4`, `172` imgs) and coastal sea (`video6_mp4`, `131` imgs) |
| **`SF14_val_train_test_N`** | 42 | 42 | **235** (`1.00%`) | **102** (`1.09%`) | 133 | `207 / 10 / 18` | `0,4` | Pre-split stock photos (`val1..20`, `train1..29`, `test1..12`): open ocean/lake waves with hands mislabeled `cls=4` |
| **`SF05_visdrone_0000`** | 32 | 57 | **175** (`0.74%`) | **73** (`0.78%`) | 102 | `168 / 7 / 0` | `1` | VisDrone aerial urban traffic & street pedestrians (`0000001_*`..`0000360_*`), 100% out-of-scope (`cls=1`) |
| **`SF09_drowning_small`** | 1 | 1 | **60** (`0.25%`) | **24** (`0.26%`) | 36 | `60 / 0 / 0` | `0,4` | Single close-up photo (`drowning_small-1-1000x573_jpg`) repeated 60 times with oversized boxes & `0<->4` collision |
| **`SF07_maritime_mvi`** | 1 | 1 | **1** (`0.004%`) | **1** (`0.01%`) | 0 | `0 / 0 / 1` | *(empty)* | Singapore Maritime Dataset frame (`MVI_1627_VIS_frame20`), open ocean cargo ships with empty `.txt` label |
| **Total** | **1,407** | **2,678** | **23,586** (`100%`) | **9,324** (`100%`) | **14,262** | `21,370 / 1,297 / 919` | `0..6` | Complete dataset accounted for |

---

## 3. Exact `SHA-256` Group Size Distribution

Why does a `23,586`-image dataset contain only `9,324` unique `image_sha256` files? Inspecting the distribution of `sha_group_size` in [`FULL_DATASET_INVENTORY.csv`](file:///E:/lifeguardDron/DrowningDetection_v1_cleaning/04_recovery_investigation/FULL_DATASET_INVENTORY.csv) reveals the exact mechanism:

| `sha_group_size` (Byte-Identical Copies) | Number of Unique `image_sha256` Groups | Total Image Files (`group_size × groups`) | Redundant Duplicate Files (`(group_size - 1) × groups`) | Forensic Explanation |
| :---: | :---: | :---: | :---: | :--- |
| **`1` (Singleton)** | **2,007** | **2,007** (`8.51%`) | 0 | Unique images exported once without byte-identical siblings |
| **`2` (Pair)** | **725** | **1,450** (`6.15%`) | **725** | Source images where 2 of 3 Roboflow export slots received identical transforms |
| **`3` (Triplet)** | **6,413** | **19,239** (`81.57%`) | **12,826** | **Primary cause:** Roboflow `"3x versions per source image"` export wrote 3 byte-identical JPEGs with different `.rf.<hash>.jpg` filenames! |
| **`4` (Quadruplet)** | **79** | **316** (`1.34%`) | **237** | Re-imported sub-datasets where the same source frame was exported across two merges |
| **`5` (Quintuplet)** | **26** | **130** (`0.55%`) | **104** | Double-exported source frames (`3 + 2` copies) across numeric and `+4`-shifted exports |
| **`6` (Sextuplet)** | **74** | **444** (`1.88%`) | **370** | Double-exported source frames (`3 + 3` copies), predominantly with `3` copies labeled `Class 0` and `3` byte-identical copies labeled `Class 4`! |
| **Total** | **9,324** | **23,586** (`100.00%`) | **14,262** (`60.47%`) | **`60.47%` of all files in `DrowningDetectionTracking_v1` are byte-for-byte identical duplicates!** |

---

## 4. Schema of `FULL_DATASET_INVENTORY.csv`

[`FULL_DATASET_INVENTORY.csv`](file:///E:/lifeguardDron/DrowningDetection_v1_cleaning/04_recovery_investigation/FULL_DATASET_INVENTORY.csv) contains `23,586` rows (sorted by `image_id` `0..23585`) with the following 20 columns:
1. `rel_image_path` — Relative path inside split archive (`<split>/images/<filename>`).
2. `filename` — Basename of the image file.
3. `original_split` — `train`, `valid`, or `test`.
4. `source_prefix` — Lowercase stem before `.rf.<hash>` (`2,678` unique).
5. `source_group` — Normalized `video_key` grouping frames of the same video (`1,407` unique).
6. `super_family` — One of the 17 exhaustive super-families (`SF01`..`SF18`).
7. `width` — Decoded image width (`640` for all `23,586`).
8. `height` — Decoded image height (`640` for all `23,586`).
9. `image_format` — File extension (`jpg` for all `23,586`).
10. `file_size` — Exact byte length of the JPEG payload.
11. `label_path` — Relative path of the paired `.txt` label file (`<split>/labels/<stem>.txt`).
12. `annotation_line_count` — Number of non-empty lines in the label file (`0` to `43`).
13. `annotation_format` — `DETECTION` (`23,570`), `MIXED_DET_POLY` (`9`), `POLYGON_ONLY` (`6`), or `EMPTY` (`1`).
14. `numeric_class_ids` — Semicolon-delimited sorted list of distinct class IDs in the file (`0`..`6`).
15. `image_sha256` — 64-character hex SHA-256 digest of the raw JPEG bytes.
16. `annotation_sha256` — 64-character hex SHA-256 digest of the raw `.txt` label bytes.
17. `sha_group_id` — Integer ID (`0..9323`) of the unique `image_sha256` equivalence class.
18. `sha_group_size` — Number of byte-identical image files sharing `sha_group_id` (`1..6`).
19. `is_exact_dup_redundant` — `0` for the single canonical representative of each `sha_group_id` (`9,324` files); `1` for redundant byte-identical copies (`14,262` files).
20. `existing_review_status` — `VERIFIED_VISUAL` (`279` directly inspected unique files across Phase 1 and Phase 2), `VERIFIED_SHA256_CLONE` (`412` byte-identical clones of directly inspected files = `691` pixel-verified files), or `PENDING_REVIEW` (`22,895` files).
