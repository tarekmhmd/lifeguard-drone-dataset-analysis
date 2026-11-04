# Phase D — Full-Dataset Annotation Forensics Report
## `DrowningDetectionTracking_v1` | Autonomous AI Lifeguard Drone (Swimming Pools Only)

**Report Date:** 2026-11-03  
**Dataset Status:** **`UNAPPROVED FOR TRAINING`**  
**Deliverables:**
- [`ANNOTATION_FORENSICS_FULL.csv`](file:///E:/lifeguardDron/DrowningDetection_v1_cleaning/04_recovery_investigation/ANNOTATION_FORENSICS_FULL.csv) (`23,586` image-level rows covering all `49,055` annotation lines)
- [`CLASS_SEMANTICS_BY_SOURCE.csv`](file:///E:/lifeguardDron/DrowningDetection_v1_cleaning/04_recovery_investigation/CLASS_SEMANTICS_BY_SOURCE.csv) (`2,147` `(super_family, video_key, class_id)` rows)
- [`visual_overlays/`](file:///E:/lifeguardDron/DrowningDetection_v1_cleaning/04_recovery_investigation/visual_overlays/) (`12` high-resolution diagnostic overlays)

---

## 1. Structural & Geometric Validity Across All `49,055` Annotation Lines

Every one of the `23,586` `.txt` label files inside `train`, `valid`, and `test` was parsed token-by-token (`01_extract_inventory.py`, cached in `boxes_raw.csv`):

| Structural Check | Count (`Lines` / `Files`) | Status | Forensic Findings |
| :--- | :---: | :---: | :--- |
| **Total Non-Empty Annotation Lines** | **49,055** across **23,585** files | Verified | `1` empty file (`0` lines): `test/labels/MVI_1627_VIS_frame20_jpg.rf.098ac3dc01adebe600fb12fc9cd1fa60.txt` |
| **Standard 5-Token Boxes (`DETECTION`)** | **49,040** across **23,579** files | Valid 5-field | `23,570` `DETECTION`-only files + `9` `MIXED_DET_POLY` files (`19` detection boxes) |
| **Multi-Vertex Polygons (`POLYGON`, >=7 odd tokens)** | **15** across **15** files | Task Mismatch | Exclusively in `train`: `6` `POLYGON_ONLY` files (`3-727-...-iStock_mp4-2_jpg`, `29` vertices / `59` tokens) + `9` `MIXED_DET_POLY` files (`...-mp4-3_jpg`, `24` vertices / `49` tokens). **All `15` are Class `4` in an out-of-scope open-water stock video.** **`0` polygon lines exist anywhere else in the dataset.** |
| **Unparseable / Non-Numeric / Even-Token Lines** | **0** | Clean | Zero syntax corruptions or NaN tokens across all `23,586` files |
| **Invalid Class IDs (outside integer `0..6`)** | **0** | Clean | All `49,055` lines have integer class IDs in `{0, 1, 2, 3, 4, 5, 6}` |
| **Out-of-Bounds Coordinates (`oob`, outside `[0,1]`)** | **0** | Clean | All `cx, cy, w, h` and polygon vertices lie within `[0.0, 1.0]` |
| **Zero or Negative Width/Height (`nonpositive_wh`)** | **0** | Clean | Minimum box area is `0.000105` (`~6.6×6.6` px at `640×640`) |
| **Oversized Full-Frame Boxes (`area > 0.90`)** | **67** boxes in **67** files (`25` unique SHAs) | Corrupted Box | `15` in `drowning_small-1-1000x573_jpg`, `12` in `drowning20_mp4`, `9` in `3-727-...-iStock_2_mp4`, etc. Box covers the entire water background rather than the person |
| **Large Boxes (`area > 0.80`)** | **422** boxes in **422** files | Suspicious | Predominantly full-frame stock-photo boxes where the whole image was boxed |
| **Tiny Boxes (`area < 0.0004`, `<13×13` px)** | **263** boxes in **149** files | VisDrone / Distant | Concentrated in `SF05_visdrone_0000` (`cls=1` distant street pedestrians) and `130000_jpg` aerial competition pools |
| **Extreme Aspect Ratio (`w/h > 8` or `h/w > 8`)** | **82** boxes in **78** files | Seam-Truncated | Caused by Mosaic/crop truncation along image borders |
| **Border-Touching Boxes (`within 0.5%` of edge)** | **11,482** boxes in **8,619** files | High | Reflects heavy random cropping, Mosaic seams, and close-up framing |
| **Synthetic Noise-Cutout Boxes (`is_noise_cutout_box`)** | **7,128** boxes in **4,913** files (`1,698` unique SHAs) | Destructive Augmentation | Target swimmer inside box is obliterated by Roboflow bounding-box salt-and-pepper speckle or solid black/gray occlusion patch |

---

## 2. The "Rosetta Stone" Forensic Discovery: Systematic `+4` and `(0,3) ↔ (4,5)` Class-Index Offset Corruption

Phase 1 identified that numeric classes `0..3` (`'3'..'6'`) and semantic classes `4..6` (`Out of Water`, `drowning`, `swimming`) were incompatible, and noted puzzling cases where people in water were labeled `Class 4 (Out of Water)`.

In Phase 2, by cross-comparing **byte-identical `image_sha256` duplicates (`195` `sha_group`s with conflicting class sets!)** and **same-video `source_prefix` / `video_key` groups (`602` prefixes)**, we uncovered the exact mathematical mechanism that corrupted the dataset's labels:

### 2.1 How the Class-Offset Corruption Happened
When `DrowningDetectionTracking_v1` (`7` classes: `0:'3', 1:'4', 2:'5', 3:'6', 4:'Out of Water', 5:'drowning', 6:'swimming'`) was assembled in Roboflow by merging multiple component exports of the **exact same underlying videos and stock photos**:
1. **1-Class Drowning Sub-Datasets (`0` ↔ `4` Shift across `276` Source Prefixes):**
   - Component datasets with a single class `0 = drowning` (`3-727-Drowning-Stock-Videos...`, `drowning-man-stock-video...`, `the-child-is-drowning-in-water...`, `child-pretending-to-be-drowning...`, `10_mov`, `video40-1..13_mp4`, `video6_mp4`, `drowning_small-1`, `istockphoto-*`, `val*`, `train*`, `test*`, `images-*`) were imported once at index `0` (`'3'`) and imported a **second time with a `+4` class-table offset** (after prepending the 4 numeric classes `0,1,2,3`), shifting original class `0 (drowning)` onto **Class `4` (`Out of Water`)**!
   - **Direct Proof on Byte-Identical Images (`phase2_sheet_04_class_offset_pairs.jpg` Tiles 13–16 & `ov05`):**
     - `train/images/The-Child-Is-Drowning-In-Water-...-iStock_mp4-0_jpg.rf.11c7e70d8250f2c6372fb62772944f1a.jpg` (`sha_group=355`) and `.rf.320a9ba0ff8851b3c18bcce233e8df06.jpg` have the **exact same `image_sha256` (`7999907c...`)** and **exact same bounding box coordinates (`0.4953125 0.5609375 0.728125 0.5875`)**, but the first file has class `0` and the second file has class `4` (`Out of Water`)!
2. **2-Class (`drowning`, `swimming`) Video Sub-Datasets (`(0,3)` ↔ `(4,5)` Shift across `drowning1/47/20/50..55_mp4`, `clipchamp-1`, `video_2022-12-07_*`):**
   - Component 2-class video datasets where original class `0 = drowning victim` (1 box per frame) and original class `1 = normal swimmer` (3–5 boxes per frame) were imported once mapped to **`(0 = '3', 3 = '6')`** and imported a second time shifted onto **`(4 = 'Out of Water', 5 = 'drowning')`**!
   - **Catastrophic Consequence (`phase2_sheet_04` Tiles 1–10 & `ov01`–`ov04`):**
     - In the `(4, 5)` copies of `drowning1_mp4`, `drowning47_mp4`, `drowning50..55_mp4`, and `Untitled-video-Made-with-Clipchamp-1-_mp4`, **Class `4` (`Out of Water`) boxes the drowning victim in the water**, while **Class `5` (`drowning`) boxes the 4–5 normal bystander swimmers**!
3. **`SF03_imageNNNN` (`1,916` Thai TV News CCTV Frames):**
   - In `461` of the `463` prefixes (`1,910` images), only a single box is present — labeled **Class `4` (`Out of Water`)** — and it boxes the **drowning child submerged inside the red TV graphic circle in the pool** (`ov08_thairath_cctv_incomplete_class4.jpg`), while leaving the two bystanders on the pool deck completely unannotated!

---

## 3. Global Semantic Audit of All 7 Classes (`0` through `6`)

| Class ID | `data.yaml` Name | Valid Boxes (`Images`) | Actual Visual Objects Observed Across Super-Families | Can This Class ID Be Trusted Globally Without Re-Annotation? |
| :---: | :---: | :---: | :--- | :---: |
| **`0`** | `'3'` | **9,890** (`8,597`) | **Mixed:** (a) Drowning/struggling victims in `drowning1/47/20/50..55_mp4`, `clipchamp-1`, `video_2022-12-07`, `stock_named`; (b) Passive/floating lane swimmers in `output_0001..0600` and `youtube-N`. | **NO** — Opaque numeric ID; conflates active drowning victims with passive/floating lane swimmers |
| **`1`** | `'4'` | **4,683** (`370`) | **3 Incompatible Domains:** (a) VisDrone urban street pedestrians/vehicles (`0000xxx`, `175` imgs); (b) Poolside dry-land people in `youtube-N` & `image724`/`1944`; (c) Aerial competition pool swimmers in `130319..130394_jpg` and dense grid boxes in `1..23_jpg`. | **NO** — Mixes urban traffic pedestrians, poolside bystanders, and in-water swimmers |
| **`2`** | `'5'` | **1,432** (`920`) | **2 Conflicting Meanings (Occurs ONLY in 4 families):** (a) In `output_0001..0600` (`973` boxes across `722` imgs) and `image724`/`1944` (`6` boxes), marks people on the **pool deck / edge**; (b) In `kakaotalk_*` (`453` boxes across `192` imgs), marks **children floating/swimming in pool water**! (`0` boxes in `youtube-*`). | **NO** — Means pool-deck person in `output_NNNN` but in-water child swimmer in `kakaotalk` |
| **`3`** | `'6'` | **9,729** (`4,737`) | **In-Water Swimmers (Pool + Ocean):** Marks active swimmers in `output_0001..0600`, `kakaotalk`, `drowning1/47/51`, `clipchamp-1`, `video_2022-12-07`, `youtube-N`, and `130473_jpg`, plus ocean beach swimmers in `3069_jpg` and `test1-5_mp4`. | **NO (Needs Remap to `swimming` on Pool-Only Subset)** — Opaque numeric ID |
| **`4`** | `'Out of Water'` | **8,160** + `15` poly (`7,756`) | **SEVERELY INVERTED (`>90%` False!):** Due to the `+4` class-index offset, in `imageNNNN` (`1,910` imgs), `drowning1/47/20/50..55_mp4`, `clipchamp-1`, `video_2022-12-07`, `stock_named`, `video40-*`, `val/train/test*`, and `other_static`, **Class `4` boxes submerged/drowning people IN WATER**! Only in a subset of `youtube-N` and `-Clipchamp-_mp4` does Class `4` mark poolside people. | **NO — CATASTROPHICALLY INVERTED** (`Out of Water` labels drowning victims in water!) |
| **`5`** | `'drowning'` | **4,409** (`2,650`) | **Corrupted by `(0,3)->(4,5)` Offset in Merged Videos:** True drowning in `output1..75` (`153` imgs); **INVERTED in `drowning47_mp4` and `Untitled-video-Made-with-Clipchamp-1-_mp4` where Class `5` boxes the 4–5 normal bystander swimmers** while Class `4` boxes the drowning child! Also appears in `youtube-N` mosaics/broadcasts. | **ONLY in Verified Clean Subsets (`output1..75`)** — Inverted in `drowning47` and `clipchamp-1`! |
| **`6`** | `'swimming'` | **10,737** (`4,511`) | **Mostly In-Water Swimmers, With Specific Contaminations:** True pool swimmers in `1..200_jpg`, clean `-Clipchamp-_mp4`, and clean `drowning1_mp4` frames; **CONTAMINATED** in `youtube-N` (boxes the **TV news PiP sign-language interpreter** at bottom-right!), `white-water_mp4` (river tubing), and `10415`/`11730_jpg` (ocean beach). | **ONLY in Pool-Verified Clean Subsets** — Must exclude `youtube-N` PiP and `white-water_mp4` |

---

## 4. Image-Level Annotation Forensic Status Breakdown (`ANNOTATION_FORENSICS_FULL.csv`)

| `annotation_forensic_status` | Raw Images (`% of 23,586`) | Unique `image_sha256` (`% of 9,324`) | Recommended Action | Forensic Description |
| :--- | :---: | :---: | :--- | :--- |
| **`CLASS_OFFSET_INVERSION_CORRUPTION`** | **6,632** (`28.12%`) | **2,657** (`28.50%`) | `KEEP_IMAGE_REANNOTATE` | Pool/water images in source groups with `0<->4` or `(0,3)<->(4,5)` class-index shift |
| **`DESTRUCTIVE_NOISE_CUTOUT_BOX`** | **4,913** (`20.83%`) | **1,698** (`18.21%`) | `EXCLUDE_DESTRUCTIVE_AUGMENTATION` | Images containing >=1 box where the swimmer is obliterated by synthetic salt-and-pepper / black cutout |
| **`OUT_OF_SCOPE_SCENE_LABELS`** | **2,642** (`11.20%`) | **1,146** (`12.29%`) | `EXCLUDE_OUT_OF_SCOPE` | Valid 5-field boxes on out-of-scope ocean, beach, river, lake, or VisDrone street scenes |
| **`UNRESOLVED_NUMERIC_CLASS_SCHEMA`** | **2,433** (`10.32%`) | **994** (`10.66%`) | `KEEP_IMAGE_REANNOTATE` | Clean pool images (`output_0001..0600`, `kakaotalk`, `130000_jpg`) using numeric IDs `0,1,2,3` |
| **`MIXED_BROADCAST_AND_NUMERIC_SCHEMA`** | **2,157** (`9.15%`) | **907** (`9.73%`) | `KEEP_IMAGE_REANNOTATE` | Non-mosaic `youtube-N` frames containing TV studio PiP interpreter (`cls=6`), dry-land demos, or numeric IDs |
| **`CANDIDATE_SEMANTIC_POOL_LABELS_UNVERIFIED`** | **2,107** (`8.93%`) | **848** (`9.09%`) | `KEEP_IMAGE_REVIEW_LABEL` | Pool images with semantic `5`/`6` labels (`output1..75`, `1..200_jpg`, clean `-Clipchamp-_mp4`) pending visual check |
| **`INCOMPLETE_AND_INVERTED_TV_CCTV_LABELS`** | **1,377** (`5.84%`) | **552** (`5.92%`) | `KEEP_IMAGE_REANNOTATE` | Uncorrupted `imageNNNN` (`THAIRATH TV 32`) frames where `cls=4` marks submerged child and bystanders are unannotated |
| **`MOSAIC_COMPOSITE_LABELS`** | **1,143** (`4.85%`) | **448** (`4.80%`) | `EXCLUDE_DESTRUCTIVE_AUGMENTATION` | Pre-baked 2×2 Mosaic composites with seam-truncated boxes and mixed tile scenes |
| **`OVERSIZED_FULL_FRAME_BOX`** | **63** (`0.27%`) | **25** (`0.27%`) | `KEEP_IMAGE_REANNOTATE` | Non-cutout images with full-frame bounding boxes covering `>90%` of image area |
| **`VERIFIED_SEMANTIC_POOL_LABELS`** | **52** (`0.22%`) | **20** (`0.21%`) | `KEEP_CANDIDATE_OR_REVIEW` | Directly visually verified pool images with clean `5 (drowning)` / `6 (swimming)` boxes (`18` unique after dedup) |
| **`AMBIGUOUS_CLASS_SEMANTICS`** | **51** (`0.22%`) | **23** (`0.25%`) | `KEEP_IMAGE_REANNOTATE` | Uncertain scene crops with semantic class IDs |
| **`MALFORMED_POLYGON_IN_DET_DATASET`** | **15** (`0.06%`) | **5** (`0.05%`) | `EXCLUDE_OUT_OF_SCOPE` | The `15` multi-vertex YOLO segmentation polygon files in `3-727-...-iStock_mp4-2` & `mp4-3` |
| **`EMPTY_ANNOTATION`** | **1** (`0.004%`) | **1** (`0.01%`) | `EXCLUDE_OUT_OF_SCOPE` | `test/images/MVI_1627_VIS_frame20_jpg...` (Singapore Maritime Dataset open-ocean vessels) |
| **Total** | **23,586** (`100%`) | **9,324** (`100%`) | — | All `23,586` files classified |

