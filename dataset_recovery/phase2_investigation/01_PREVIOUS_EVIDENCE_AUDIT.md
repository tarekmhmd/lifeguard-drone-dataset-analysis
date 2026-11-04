# Phase A — Previous Evidence Audit Report
## `DrowningDetectionTracking_v1` | Autonomous AI Lifeguard Drone (Swimming Pools Only)

**Audit Date:** 2026-11-03  
**Dataset Status:** **`UNAPPROVED FOR TRAINING`**  
**Access Mode:** Strict Read-Only Forensics (Original archives, images, labels, splits, and prior deliverables untouched)

---

## 1. Executive Summary of the Phase A Audit

Before inspecting the unreviewed images across `DrowningDetectionTracking_v1`, all five Phase 1 investigation deliverables in [`03_source_investigation/`](file:///E:/lifeguardDron/DrowningDetection_v1_cleaning/03_source_investigation/) and all supporting baseline audit artifacts in [`DrowningDetectionTracking_v1_audit_results/`](file:///E:/lifeguardDron/DrowningDetectionTracking_v1_audit_results/) were cross-audited against an exhaustive, deterministic read-only pass over all **23,586 images** and **49,055 annotation lines** inside the three original split archives (`train`, `valid`, `test`).

### Key Audit Findings
1. **Image, Split, and Structural Annotation Totals Are 100% Confirmed:**
   - Total dataset images: **`23,586`** (`train = 21,369`, `valid = 1,215`, `test = 1,002`), all `640×640` JPEG (`100%` decode success, `0` corrupt image files, `0` missing `.txt` label files, `0` orphan label files).
   - Total annotation lines: **`49,055`**, consisting of **`49,040` valid 5-field YOLO detection bounding boxes** and **`15` multi-vertex YOLO segmentation polygon lines** (in `15` files of `3-727-...-iStock_mp4-2_jpg` and `mp4-3_jpg`), plus **`1` empty label file** (`test/labels/MVI_1627_VIS_frame20_jpg.rf.098ac3dc01adebe600fb12fc9cd1fa60.txt`).
2. **Phase 1 Visual Inspection Coverage Was `106` Rows (`106` Unique Inspected Tiles = `0.45%` of Raw Dataset):**
   - [`VISION_REVIEW_RESULTS.csv`](file:///E:/lifeguardDron/DrowningDetection_v1_cleaning/03_source_investigation/VISION_REVIEW_RESULTS.csv) contains **`106` rows** across Contact Sheets 1–5, polygon overlays, empty-label review, and cross-split leakage pairs, leaving **`23,480` raw images** uninspected at the end of Phase 1.
   - **Phase A Filename Audit of `VISION_REVIEW_RESULTS.csv` (`80 + 14 + 12 = 106`):** Cross-checking `VISION_REVIEW_RESULTS.csv` against the exact archive entries revealed that `80` rows match `rel_image_path` verbatim; `14` rows (`invalid_annotations` rows 55–68) recorded only the first 8 hex characters of `.rf.<hash>` shown on the overlay banner (`30c52f25...`, `b9566e32...`, etc., mapping 1-to-1 to the `14` remaining files in `invalid_annotations.csv`); and `12` rows (`class_semantics_extra` Contact Sheet 4 rows 70–81) had placeholder `.rf.` hashes because the Contact Sheet 4 tile banner printed only `<source_prefix>` (`124_png_jpg`, `134_png_jpg`, `image1944_jpg`, etc.). All `106` rows were resolved to their exact archive paths in Phase 2.
3. **Critical Discrepancies Identified and Resolved in Prior Reports:**
   - **Discrepancy 1 — Massive Unreported Within-Split Exact Image Duplication (`14,262` redundant byte-identical files):** The baseline audit file [`duplicate_images.csv`](file:///E:/lifeguardDron/DrowningDetectionTracking_v1_audit_results/duplicate_images.csv) (`272` rows) **only recorded cross-split hash groups** (`99` cross-split `exact_image_hash` groups covering `365` files, and `173` cross-split `exact_label_hash` groups covering `702` files). Full-dataset `SHA-256` hashing of all `23,586` image payloads proves there are only **`9,324` unique `image_sha256` byte streams** in the entire dataset (`2,007` singletons + **`7,317` multi-file `SHA-256` duplicate groups covering `21,579` files**, i.e., **`14,262` redundant byte-identical copies** or **`60.47%`** of the dataset!).
   - **Discrepancy 2 — Arithmetic Error in `audit_summary.txt` & Per-Class Box Count Mismatches in Phase 1 `VISION_CLASS_SEMANTICS_REPORT.md`:** [`audit_summary.txt`](file:///E:/lifeguardDron/DrowningDetectionTracking_v1_audit_results/audit_summary.txt) stated `Classes 0-3: 23,238 boxes (47.4%); classes 4-6: 25,802 boxes (52.6%)`, whereas the exact sum from [`class_distribution.csv`](file:///E:/lifeguardDron/DrowningDetectionTracking_v1_audit_results/class_distribution.csv) and `boxes_raw.csv` is **Classes `0–3`: `25,734` boxes (`52.48%`)** and **Classes `4–6`: `23,306` boxes (`47.52%`)**. In addition, Phase 1 [`VISION_CLASS_SEMANTICS_REPORT.md`](file:///E:/lifeguardDron/DrowningDetection_v1_cleaning/03_source_investigation/VISION_CLASS_SEMANTICS_REPORT.md) reported a swapped/approximate per-class box count column and mistakenly cited a Class-`2` box on `youtube-4_jpg` (whereas `boxes_raw.csv` proves Class `2` has `0` boxes in `youtube-*` and occurs exclusively in `output_*` [`973` boxes], `kakaotalk_*` [`453` boxes], `image724_jpg` [`3` boxes], and `image1944_jpg` [`3` boxes]).
   - **Discrepancy 3 — Source-Prefix Grouping Rule in Phase 1 `VISION_DUPLICATE_AND_LEAKAGE_REPORT.md`:** [`SOURCE_GROUP_INVENTORY.csv`](file:///E:/lifeguardDron/DrowningDetection_v1_cleaning/03_source_investigation/SOURCE_GROUP_INVENTORY.csv) and [`same_source_prefix.csv`](file:///E:/lifeguardDron/DrowningDetectionTracking_v1_audit_results/same_source_prefix.csv) define **`2,678` unique lowercase `.rf.` prefixes**, of which **`685` prefixes (`12,667` images, `53.71%`) span multiple splits**. A section of Phase 1 [`VISION_DUPLICATE_AND_LEAKAGE_REPORT.md`](file:///E:/lifeguardDron/DrowningDetection_v1_cleaning/03_source_investigation/VISION_DUPLICATE_AND_LEAKAGE_REPORT.md) cited `1,616` of `4,949` groups due to splitting filenames on `-` instead of `.rf.`.

---

## 2. Inventory & Split Reconciliation Table

| Split | ZIP Archive Entries | Images (`/images/*.jpg`) | Labels (`/labels/*.txt`) | Valid 5-Field Boxes (`DETECTION`) | Polygon Lines (`POLYGON`, >=7 tokens) | Unparseable Lines | Empty Label Files (`0` bytes) | Unique `image_sha256` Hashes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`train`** | 42,738 | **21,369** | **21,369** | **44,834** | **15** | 0 | 0 | **8,348** |
| **`valid`** | 2,430 | **1,215** | **1,215** | **2,655** | 0 | 0 | 0 | **826** |
| **`test`** | 2,004 | **1,002** | **1,002** | **1,551** | 0 | 0 | **1** | **251** |
| **Total** | **47,172** | **23,586** | **23,586** | **49,040** | **15** | **0** | **1** | **9,324** *(net across dataset)* |

*Note:* The sum of per-split unique `image_sha256` hashes (`8,348 + 826 + 251 = 9,425`) exceeds the global unique `image_sha256` total (`9,324`) by `101` because **`99` exact `image_sha256` duplicate groups** (`97` spanning 2 splits and `2` spanning or distributed across pairs: `58` `train+valid`, `30` `train+test`, `11` `valid+test`) appear in more than one split.

---

## 3. Exact Per-Class Annotation Reconciliation (`49,040` Valid Boxes + `15` Polygons)

The table below reconciles [`data.yaml`](file:///E:/lifeguardDron/DrowningDetectionTracking_v1/data.yaml), [`class_distribution.csv`](file:///E:/lifeguardDron/DrowningDetectionTracking_v1_audit_results/class_distribution.csv), [`audit_summary.txt`](file:///E:/lifeguardDron/DrowningDetectionTracking_v1_audit_results/audit_summary.txt), and our full-dataset line-by-line parse (`boxes_raw.csv`):

| Class ID | `data.yaml` Name | `train` Boxes (`Images`) | `valid` Boxes (`Images`) | `test` Boxes (`Images`) | Total Valid 5-Field Boxes | Total Images w/ Valid Box | Polygon Lines (`train`) | Phase 1 Report Discrepancy Corrected |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`0`** | `'3'` | 9,032 (`7,856`) | 546 (`466`) | 312 (`275`) | **9,890** (`20.17%`) | **8,597** | 0 | Phase 1 table listed `10,737` (swapped with Class 6) |
| **`1`** | `'4'` | 4,111 (`342`) | 442 (`19`) | 130 (`9`) | **4,683** (`9.55%`) | **370** | 0 | Phase 1 table listed `9,890` (swapped with Class 0) |
| **`2`** | `'5'` | 1,322 (`850`) | 84 (`54`) | 26 (`16`) | **1,432** (`2.92%`) | **920** | 0 | Phase 1 table listed `9,729`; occurs **only** in `output_*`, `kakaotalk_*`, `image724`, `image1944` (`0` in `youtube-*`) |
| **`3`** | `'6'` | 8,773 (`4,218`) | 583 (`276`) | 373 (`243`) | **9,729** (`19.84%`) | **4,737** | 0 | Phase 1 table listed `8,160` (swapped with Class 4) |
| **`4`** | `'Out of Water'` | 7,375 (`6,978`) | 387 (`384`) | 398 (`388`) | **8,160** (`16.64%`) | **7,750** *(+6 poly-only = 7,756)* | **15** *(across 15 imgs)* | Phase 1 table listed `4,683`; suffers from **systematic `+4` class-index offset** |
| **`5`** | `'drowning'` | 4,123 (`2,434`) | 172 (`113`) | 114 (`103`) | **4,409** (`8.99%`) | **2,650** | 0 | Confirmed exact (`4,409` boxes); suffers from `+2` offset (`3->5`) in merged videos |
| **`6`** | `'swimming'` | 10,098 (`4,234`) | 441 (`186`) | 198 (`91`) | **10,737** (`21.89%`) | **4,511** | 0 | Phase 1 table listed `1,432` (swapped with Class 2) |
| **Subtotal `0–3`** | Numeric (`'3'..'6'`) | 23,238 | 1,655 | 841 | **25,734** (`52.48%`) | 12,866 | 0 | `audit_summary.txt` mistakenly used `train`-only `23,238` as the dataset total |
| **Subtotal `4–6`** | Semantic names | 21,596 | 1,000 | 710 | **23,306** (`47.52%`) | 13,581 | 15 | `audit_summary.txt` subtracted `23,238` from `49,040` to get `25,802` |
| **Grand Total** | **All 7 Classes** | **44,834** | **2,655** | **1,551** | **49,040** (`100.00%`) | **23,579** *(+6 poly-only + 1 empty = 23,586)* | **15** | All `49,055` lines accounted for with zero discrepancy |

---

## 4. Audit of Supporting Evidence Files & Separation of Duplicate/Leakage Categories

Per Section 4 of the Master Execution Prompt, we audited [`invalid_annotations.csv`](file:///E:/lifeguardDron/DrowningDetectionTracking_v1_audit_results/invalid_annotations.csv), [`duplicate_images.csv`](file:///E:/lifeguardDron/DrowningDetectionTracking_v1_audit_results/duplicate_images.csv), [`same_source_prefix.csv`](file:///E:/lifeguardDron/DrowningDetectionTracking_v1_audit_results/same_source_prefix.csv), [`SCENE_CLASSIFICATION_PROGRESS.csv`](file:///E:/lifeguardDron/DrowningDetection_v1_cleaning/03_source_investigation/SCENE_CLASSIFICATION_PROGRESS.csv), and [`SOURCE_GROUP_INVENTORY.csv`](file:///E:/lifeguardDron/DrowningDetection_v1_cleaning/03_source_investigation/SOURCE_GROUP_INVENTORY.csv), and separated all four distinct duplication/leakage mechanisms:

### 4.1 `invalid_annotations.csv` (15 Rows) — Verified 100% Complete
- Contains `15` rows in `train/labels/` across `3-727-Drowning-Stock-Videos-and-Royalty-Free-Footage-iStock_mp4-2_jpg` (`6` files, `29` vertices / `59` tokens each) and `...-mp4-3_jpg` (`9` files, `24` vertices / `49` tokens each).
- Our full-dataset scan of all `49,055` lines across `23,586` label files confirms **`0` additional polygon lines**, **`0` non-integer class IDs**, **`0` out-of-range class IDs**, **`0` out-of-bounds `[0,1]` coordinates**, and **`0` non-positive width/height boxes** anywhere else in `train`, `valid`, or `test`.

### 4.2 Strict Separation of the Four Duplication & Leakage Categories
Prior reports partially conflated image-hash duplicates, label-hash duplicates, augmented siblings, and video-frame continuity. Below is the exact, non-overlapping forensic breakdown across all `23,586` images:

| Evidence Category | Scope | Distinct Groups | Total Files Involved | Redundant / Leaking Files | Cross-Split Groups (`Files`) | Within-Split Groups (`Files`) | Forensic Root Cause |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. Exact Image Duplicates (`image_sha256`)** | Identical JPEG byte stream | **7,317** *(of 9,324 unique SHAs)* | **21,579** (`91.49%`) | **14,262** (`60.47%`) | **99** (`365` files) | **7,218** (`21,214` files) | Roboflow `"3x versions per source image"` export wrote 3 byte-identical JPEGs (`6,413` groups of size 3) + re-uploaded sub-datasets (`size 4..6`). `duplicate_images.csv` reported only the `99` cross-split groups! |
| **2. Exact Label Duplicates (`label_sha256`)** | Identical `.txt` byte stream | **6,912** *(multi-file label hashes)* | **20,165** | **13,253** | **173** (`702` files in `duplicate_images.csv`) | **6,739** (`19,463` files) | Includes both byte-identical images with identical `.txt` files AND distinct/cropped images that happen to share identical box coordinates (`74` of the `173` cross-split label groups have different `image_sha256`!). |
| **3. Conflicting Labels on Byte-Identical Images** | Same `image_sha256`, **different** `label_sha256` | **284** *(label variants)* / **215** *(different class sets!)* | **1,089** | **805** | **41** | **243** | **The Rosetta Stone of `DrowningDetectionTracking_v1`:** Identical JPEG files imported twice with a `+4` / `(0,3)->(4,5)` class-index shift (`0` vs `4`, or `0;3` vs `4;5`)! |
| **4. Perceptual Near-Duplicates (`pHash`/`dHash` on unique SHAs)** | Distinct `image_sha256`, `min(pHash, pHash_flip) <= 6` | **39,899 pairs** *(across 9,324 unique SHAs)* | **5,842** unique SHAs | — | **2,114** cross-split pairs (`min_dist <= 6`) | **37,785** within-split pairs | Horizontal flips (`9,641` pairs at `min_dist == 0`), brightness/contrast shifts, minor crops, and adjacent 30-fps video frames. |
| **5. Same `.rf.` Source Prefix (`source_prefix`)** | Shared `<stem>.rf.<hash>` prefix | **2,678** total (`2,219` multi-file) | **23,127** *(in multi-file prefixes)* | — | **685** prefixes (**`12,667` files**, `53.71%`) | **1,534** multi-file prefixes (`10,460` files) | Multiple Roboflow augmentation/export passes from the same source frame or image stem (`same_source_prefix.csv`). |
| **6. Same Underlying Video (`atomic_video_unit`)** | Shared source video clip across prefixes | **1,013** total (`322` cross-split) | **23,586** | — | **322** units (**`16,498` files**, **`69.95%`**) | **691** single-split units (`7,088` files) | Consecutive frames extracted from the same MP4/MOV/CCTV broadcast (e.g., `output_0001..0600` = `2,276` imgs; `image1..3763` = `1,916` imgs; `drowning1/47/50..55_mp4` = `2,265` imgs) split randomly across `train`/`valid`/`test`. |

---

## 5. Audit of `SCENE_CLASSIFICATION_PROGRESS.csv`

- [`SCENE_CLASSIFICATION_PROGRESS.csv`](file:///E:/lifeguardDron/DrowningDetection_v1_cleaning/03_source_investigation/SCENE_CLASSIFICATION_PROGRESS.csv) in `03_source_investigation/` tracked `20` high-level pattern groups covering `67` unique visually inspected images from Phase 1 Contact Sheets 1–3 (`36` tiles) plus `31` additional tiles from Sheets 4–5 and duplicate checks (`106` total rows in `VISION_REVIEW_RESULTS.csv`).
- It explicitly recorded `23,480` images as uninspected.
- In Phase 2, we expanded direct visual inspection by generating and visually inspecting **12 new 4×4 contact sheets (`192` tiles = `183` new unique dataset images with zero overlap against Phase 1)** across all **17 exhaustive super-families** (`289` directly visually inspected unique files in total, which directly verify the exact pixel bytes of **`775` files** via `image_sha256` identity), while keeping all remaining `22,811` uninspected files strictly marked with `reviewer_status = PENDING_REVIEW` in [`SCENE_CLASSIFICATION_FULL.csv`](file:///E:/lifeguardDron/DrowningDetection_v1_cleaning/04_recovery_investigation/SCENE_CLASSIFICATION_FULL.csv).
