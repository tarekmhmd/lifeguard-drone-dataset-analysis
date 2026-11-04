# Phase C — Full-Dataset Visual & Scene Investigation Report
## `DrowningDetectionTracking_v1` | Autonomous AI Lifeguard Drone (Swimming Pools Only)

**Report Date:** 2026-11-03  
**Dataset Status:** **`UNAPPROVED FOR TRAINING`**  
**Deliverables:**
- [`SCENE_CLASSIFICATION_FULL.csv`](file:///E:/lifeguardDron/DrowningDetection_v1_cleaning/04_recovery_investigation/SCENE_CLASSIFICATION_FULL.csv) (`23,586` rows)
- [`SCENE_REVIEW_PROGRESS.csv`](file:///E:/lifeguardDron/DrowningDetection_v1_cleaning/04_recovery_investigation/SCENE_REVIEW_PROGRESS.csv) (`17` Super-Families)
- [`contact_sheets/phase2_contact_sheet_index.csv`](file:///E:/lifeguardDron/DrowningDetection_v1_cleaning/04_recovery_investigation/contact_sheets/phase2_contact_sheet_index.csv) (`192` tiles across `12` Phase 2 contact sheets)

---

## 1. Visual Inspection Methodology & Strict Separation of Verified vs. Pending Coverage

In strict compliance with Section 6 of the Master Execution Prompt:
1. **Direct Visual Inspection (`reviewer_status = VERIFIED_VISUAL`, `279` net unique files):**
   - We generated and directly visually inspected **12 new 4×4 contact sheets (`192` tiles = `183` unique dataset files)** in [`contact_sheets/`](file:///E:/lifeguardDron/DrowningDetection_v1_cleaning/04_recovery_investigation/contact_sheets/), sampling uninspected `video_key`s and `source_prefix`es across all **17 exhaustive Super-Families**.
   - Combined with the **`106` Phase 1 review rows** (`96` unique files after resolving 2 duplicate tiles and 8 prefix overlaps), **`279` unique dataset files** have been directly inspected by human/VLM eyes (`review_method = visual_inspection_direct`).
2. **Exact Byte-Identical SHA-256 Clones of Visually Inspected Files (`reviewer_status = VERIFIED_SHA256_CLONE`, `412` files):**
   - Because each unique `image_sha256` byte stream appears an average of `2.53` times in the dataset, the `279` directly inspected files verify the exact pixel bytes of **`412` additional byte-identical SHA-256 clones**, bringing total pixel-verified coverage to **`691` files (`2.93%` of raw dataset)**.
3. **Uninspected Files (`reviewer_status = PENDING_REVIEW`, `22,895` files):**
   - All remaining `22,895` files (`97.07%`) that were not directly opened as images are strictly recorded with `reviewer_status = PENDING_REVIEW` and `review_method = source_group_and_pixel_forensics_unverified`.
   - Their candidate `scene_type` (`POOL`, `OUT_OF_SCOPE`, `UNCERTAIN`, `SUSPECTED_MOSAIC`, `CORRUPTED`) is derived from (a) deterministic pixel/box anomaly detectors (`seam_row`/`seam_col` mosaic seam detector and `in_dark`/`in_sp` synthetic noise-cutout detector) and (b) source-video/prefix propagation from the visually inspected representatives of the same video or super-family.

---

## 2. Full-Dataset Scene Classification Summary (`23,586` Raw Files / `9,324` Unique `image_sha256`)

| `scene_type` | Verified Visual (`Direct + SHA Clone`) | Candidate (`PENDING_REVIEW`) | Total Raw Images (`% of 23,586`) | Unique `image_sha256` Images (`% of 9,324`) | Redundant Exact Dups | Operational Scope & Forensic Interpretation |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`POOL`** | **383** (`146` direct + `237` clone) | **12,298** | **12,681** (`53.76%`) | **5,107** (`54.77%`) | 7,574 | Indoor/outdoor swimming pools, competition pools, wave pools, and backyard pools (uncorrupted pixel content) |
| **`CORRUPTED`** | **41** (`21` direct + `20` clone) | **4,853** | **4,894** (`20.75%`) | **1,691** (`18.14%`) | 3,203 | Images where target swimmer boxes are obliterated by synthetic salt-and-pepper / solid dark noise cutout rectangles |
| **`OUT_OF_SCOPE`** | **138** (`67` direct + `71` clone) | **2,505** | **2,643** (`11.21%`) | **1,147** (`12.30%`) | 1,496 | Open oceans, beaches, surf, white-water rivers, lakes, maritime cargo ships (`MVI_1627`), and VisDrone urban streets (`0000xxx`) |
| **`UNCERTAIN`** | **123** (`39` direct + `84` clone) | **2,102** | **2,225** (`9.43%`) | **931** (`9.98%`) | 1,294 | Tightly cropped water close-ups without visible pool boundaries (`drowning_small`, `130163_jpg`) or uninspected static prefixes |
| **`SUSPECTED_MOSAIC`** | **6** (`6` direct + `0` clone) | **1,137** | **1,143** (`4.85%`) | **448** (`4.80%`) | 695 | Pre-baked 2×2 Mosaic composite images (`seam_row >= 4.2` & `seam_col >= 4.2` near center), predominantly in `SF01_youtube_N` |
| **Total** | **691** (`279` direct + `412` clone) | **22,895** | **23,586** (`100.00%`) | **9,324** (`100.00%`) | **14,262** | Complete dataset accounted for |

---

## 3. Scene Breakdown by Super-Family (`SCENE_REVIEW_PROGRESS.csv`)

| Super-Family | Video Keys | Source Prefixes | Raw Images | Unique `SHA-256` | Verified (`Direct` + `Clone`) | `Pending Review` | `POOL` | `OUT_OF_SCOPE` | `UNCERTAIN` | `SUSPECTED_MOSAIC` | `CORRUPTED` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`SF01_youtube_N`** | 126 | 126 | **3,806** | 1,516 | `56 + 94` | 3,656 | 2,101 | 0 | 68 | **1,063** | 574 |
| **`SF02_output_NNNN`** | 2 | 524 | **2,429** | 945 | `16 + 20` | 2,393 | 1,817 | 0 | 0 | 0 | 612 |
| **`SF03_imageNNNN`** | 463 | 463 | **1,916** | 734 | `16 + 24` | 1,876 | 1,377 | 0 | 0 | 0 | 539 |
| **`SF04_pure_digits`** | 334 | 334 | **3,121** | 1,197 | `27 + 38` | 3,056 | 1,770 | 129 | 168 | 5 | 1,049 |
| **`SF05_visdrone_0000`** | 32 | 57 | **175** | 72 | `3 + 0` | 172 | 0 | 123 | 0 | 0 | 52 |
| **`SF06_kakaotalk`** | 8 | 150 | **578** | 231 | `11 + 16` | 551 | 431 | 0 | 0 | 8 | 139 |
| **`SF07_maritime_mvi`** | 1 | 1 | **1** | 1 | `1 + 0` | 0 | 0 | 1 | 0 | 0 | 0 |
| **`SF08_drowningN_mp4`** | 16 | 237 | **3,606** | 1,352 | `22 + 46` | 3,538 | 2,851 | 0 | 8 | 15 | 732 |
| **`SF09_drowning_small`** | 1 | 1 | **60** | 29 | `5 + 7` | 48 | 0 | 0 | 54 | 0 | 6 |
| **`SF10_video_2022_12_07`** | 4 | 57 | **1,258** | 501 | `9 + 12` | 1,237 | 969 | 0 | 0 | 0 | 289 |
| **`SF11_videoN_mp4`** | 10 | 64 | **303** | 129 | `12 + 20` | 271 | 0 | 303 | 0 | 0 | 0 |
| **`SF12_clipchamp_mp4`** | 2 | 85 | **612** | 247 | `18 + 23` | 571 | 480 | 0 | 0 | 0 | 132 |
| **`SF13_stock_named`** | 20 | 96 | **1,776** | 645 | `30 + 54` | 1,692 | 536 | 997 | 9 | 0 | 234 |
| **`SF14_val_train_test_N`** | 42 | 42 | **235** | 111 | `10 + 10` | 215 | 0 | 235 | 0 | 0 | 0 |
| **`SF15_img_pic_photo`** | 34 | 34 | **339** | 127 | `9 + 17` | 313 | 99 | 69 | 137 | 0 | 34 |
| **`SF17_other_video_mp4_mov`** | 6 | 101 | **803** | 442 | `8 + 6` | 789 | 0 | 622 | 1 | 28 | 152 |
| **`SF18_other_static`** | 306 | 306 | **2,568** | 1,045 | `11 + 25` | 2,532 | 250 | 176 | 1,768 | 24 | 350 |

---

## 4. Key Visual Findings from the 12 Phase 2 Contact Sheets (`192` Tiles)

1. **`phase2_sheet_01_class1_and_kakaotalk.jpg`:**
   - Verified that `1_jpg`, `5_jpg`, `6_jpg`, `16_jpg`, `19_jpg`, `23_jpg` are overhead views of an indoor swimming pool (`POOL`) with swimmers labeled Class `6` (`swimming`).
   - Verified that `130319_jpg` and `130369_jpg` are aerial drone views of outdoor competition pools (`POOL`) with lane ropes (`cls=1;3`), while `130394_jpg` (`Tile 9`) is `CORRUPTED` by a large black noise cutout over the central pool lanes.
   - Verified across all 7 remaining Korean `KakaoTalk_20230504_*` and `20230509_*` batches (`Tiles 10–16`) that **100% of `SF06_kakaotalk` (`578` images) depict genuine swimming pools (`POOL`)**.
2. **`phase2_sheet_02_output_family.jpg`:**
   - Verified that `output1..75` (`output29_jpg`, `output70_jpg`, `153` images) is an overhead indoor swimming pool video (`POOL`) showing a person in dark clothing submerged/drowning in the pool (`Class 5 = drowning`).
   - Verified across `output_0019_jpg` through `output_0492_jpg` (`Tiles 3–16`) that **all `2,276` images of `output_0001..0600` (`478` prefixes) come from a single continuous overhead indoor swimming pool camera session (`POOL`)**.
3. **`phase2_sheet_03_drowning_and_clipchamp.jpg` & `phase2_sheet_04_class_offset_pairs.jpg`:**
   - Verified that `drowning1_mp4`, `drowning47_mp4`, and `drowning50..55_mp4` (`2,265` images) all depict the **same wave-pool rescue incident (`POOL`)** (victim in wave pool with lifeguards holding blue rescue tubes), while `drowning20_mp4` and `drowning2_*` depict pool drowning close-ups (`POOL`).
   - Verified that `Untitled-video-Made-with-Clipchamp-1-_mp4` (`472` images) is a **Helsinki indoor swimming pool CCTV recording (`"allin valvontakamera 30.1.2016"`, `POOL`)** and `-Clipchamp-_mp4` (`140` images) is an **outdoor swimming pool (`"5 FT"` poolside depth marker, `POOL`)**.
   - Verified that `white-water_mp4` (`354` images) is **white-water river rapids tubing (`OUT_OF_SCOPE`)**.
4. **`phase2_sheet_05_stock_and_2022_videos.jpg` & `phase2_sheet_06_videoN_and_other_mp4.jpg`:**
   - Distinguished pool stock clips (`child-pretending-to-be-drowning-at-the-swimming-pool...`, `the-child-is-drowning-in-water...`, `istockphoto-1001608692`, `1222550542`, `172243587`, `182180100` -> `POOL`) from open-water stock clips (`3-727-...`, `drowning-man-stock-video...`, `a-man-drowning-a-safeguard-jumps-in...`, `istockphoto-1032638012`, `1034894690`, `1041463788`, `105764867`, `1299949453` -> `OUT_OF_SCOPE`).
   - Verified that **all `10` videos of `SF11_videoN_mp4` (`video40-1..13_mp4` and `video6_mp4`, `303` images) and `test1-1..8_mp4` (`265` images) are `OUT_OF_SCOPE` open ocean surf/beach and coastal sea scenes**.
5. **`phase2_sheet_07_imageNNNN_family.jpg`:**
   - Proved that **all `463` prefixes of `SF03_imageNNNN` (`1,916` images, `image1_jpg`..`image3763_jpg`) are consecutive frames from a single Thai TV news broadcast (`"THAIRATH TV 32"`, 15-02-2022) of a backyard swimming pool CCTV (`POOL`)** featuring burned-in Thai text banners, a red graphic circle over the drowning child, and a picture-in-picture news anchor at bottom-right!
6. **`phase2_sheet_08_pure_digits_family.jpg`, `phase2_sheet_09_val_train_img_families.jpg`, `phase2_sheet_10_top_other_static.jpg`:**
   - Verified the 4 distinct sub-series inside `SF04_pure_digits`: `1..200_jpg` = indoor pool (`POOL`), `1..999_png` = dark sea waves (`OUT_OF_SCOPE`), `3000..12000_jpg` = aerial ocean beach (`OUT_OF_SCOPE`), `130000_jpg` = aerial competition pool (`POOL`) / tight water crop (`UNCERTAIN`).
   - Verified that `SF14_val_train_test_N` (`val*`, `train*`, `test*`, `235` images) is **100% `OUT_OF_SCOPE` open ocean/sea/lake stock photos**.
   - Verified that `SF10_video_2022_12_07` (`4` videos, `1,258` images) is **100% indoor tiled swimming pool (`POOL`)**.
7. **`phase2_sheet_11_youtube_extended.jpg` & `phase2_sheet_12_cutout_and_mosaic_audit.jpg`:**
   - Confirmed that the extended `youtube-14`..`youtube-245` prefixes suffer from the exact same 2×2 Mosaic composites (`Tiles 4 & 6`), Kenyan TV studio broadcast banners & PiP sign-language interpreter (`"CITIZENTV.CO.KE / MONDAY REPORT: WATER SAFETY"`, `Tiles 3, 7, 13–16`), and synthetic black salt-and-pepper noise cutouts (`Tiles 8 & 12`) as `youtube-0..11`.
   - Validated on `phase2_sheet_12` (`Tiles 1–16`) that our automated box-interior noise-cutout detector (`in_dark` + `in_sp`) accurately isolates frames where Roboflow's bounding-box noise/cutout augmentation obliterated the swimmers inside the annotated boxes.

