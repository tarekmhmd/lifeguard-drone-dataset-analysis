# Phase F — Recovery & Annotation Retention Assessment Report
## `DrowningDetectionTracking_v1` | Autonomous AI Lifeguard Drone (Swimming Pools Only)

**Report Date:** 2026-11-03  
**Dataset Status:** **`UNAPPROVED FOR TRAINING`**  
**Manifest Deliverable:** [`RECOVERY_CANDIDATES.csv`](file:///E:/lifeguardDron/DrowningDetection_v1_cleaning/04_recovery_investigation/RECOVERY_CANDIDATES.csv) (`23,586` rows × `15` columns)

---

## 1. Strict Recovery Decision Rules Enforced (Section 9 Compliance)

Every one of the `23,586` images in [`RECOVERY_CANDIDATES.csv`](file:///E:/lifeguardDron/DrowningDetection_v1_cleaning/04_recovery_investigation/RECOVERY_CANDIDATES.csv) was assigned one of the 10 permitted recovery decisions under the following mandatory forensic constraints:

1. **Exact Duplicate Rule (`EXCLUDE_DUPLICATE`):** Only **one** canonical representative of each of the `9,324` unique `image_sha256` groups may be considered for retention. All **`14,262` redundant byte-identical copies (`60.47%` of raw dataset)** are strictly assigned **`EXCLUDE_DUPLICATE`**.
2. **Operational Scope Rule (`EXCLUDE_OUT_OF_SCOPE` & `EXCLUDE_WRONG_OBJECT`):** All unique images depicting beaches, open oceans, surf, white-water rivers, lakes, or maritime shipping are assigned **`EXCLUDE_OUT_OF_SCOPE`** (`1,094` unique SHAs). All unique VisDrone urban street/traffic images (`SF05_visdrone_0000`) are assigned **`EXCLUDE_WRONG_OBJECT`** (`53` non-cutout unique SHAs).
3. **Destructive Augmentation Rule (`EXCLUDE_DESTRUCTIVE_AUGMENTATION`):** All unique images containing 2×2 Mosaic composites (`448` unique SHAs) or synthetic salt-and-pepper / solid dark noise-cutout boxes (`1,698` unique SHAs) are assigned **`EXCLUDE_DESTRUCTIVE_AUGMENTATION`** (`2,146` unique SHAs).
4. **Strict `KEEP_CANDIDATE` Gate:** An image may be marked **`KEEP_CANDIDATE`** (`image_recoverable = YES`, `annotation_usable_as_is = YES_VERIFIED`) **only** if:
   - It is the single canonical representative of its `image_sha256` group (`is_sha_representative == 1`),
   - Its swimming-pool scene (`POOL`) has been **directly visually verified** (`reviewer_status` in `VERIFIED_VISUAL` or `VERIFIED_SHA256_CLONE`), and
   - Its bounding boxes use verified clean semantic classes (`5 = drowning` and/or `6 = swimming`) with **zero** class-offset collision, **zero** noise cutouts, and **zero** oversized boxes (`18` unique verified images).
5. **Unverified Clean Semantic Pool Images (`KEEP_IMAGE_REVIEW_LABEL`):** Unique swimming-pool images (`is_sha_representative == 1`) from verified pool families (`output1..75`, `1..200_jpg`, clean `-Clipchamp-_mp4`) that carry clean semantic classes (`5` or `6`) without offset collision, but whose individual frame has `reviewer_status = PENDING_REVIEW`, are assigned **`KEEP_IMAGE_REVIEW_LABEL`** (`737` unique SHAs).
6. **Recoverable Pool Images With Corrupted / Numeric / Offset Labels (`KEEP_IMAGE_REANNOTATE`):** Unique swimming-pool images (`is_sha_representative == 1`) whose pixel content is clean (`POOL`), but whose `.txt` annotations suffer from `(0,3) ↔ (4,5)` / `0 ↔ 4` class-offset inversion, incomplete TV CCTV `Class 4` boxes (`imageNNNN`), oversized `>90%` boxes, or opaque numeric class IDs (`0, 1, 2, 3` in `output_0001..0600`, `kakaotalk`, `130000_jpg`), are assigned **`KEEP_IMAGE_REANNOTATE`** (`3,463` unique SHAs).
7. **Ambiguous / Mixed-Broadcast Images (`HUMAN_REVIEW_REQUIRED`):** Unique non-mosaic, non-cutout images in `SF01_youtube_N` (contaminated with Kenyan TV broadcast studio graphics, PiP sign-language interpreter, and dry-land rescue demos, `882` unique SHAs) or with `UNCERTAIN` scene classification (`931` unique SHAs) are assigned **`HUMAN_REVIEW_REQUIRED`** (`1,813` unique SHAs).

---

## 2. Full-Dataset Recovery Decision Breakdown (`23,586` Raw Files / `9,324` Unique `image_sha256`)

| `recovery_decision` | Raw Files (`% of 23,586`) | Unique `image_sha256` (`% of 9,324`) | `image_recoverable` | `annotation_usable_as_is` | Action Required Before Training |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`EXCLUDE_DUPLICATE`** | **14,262** (`60.47%`) | **0** *(redundant copies)* | `NO` | `NO` | Drop immediately (`14,262` byte-identical `SHA-256` clones) |
| **`KEEP_IMAGE_REANNOTATE`** | **3,463** (`14.68%`) | **3,463** (`37.14%`) | **`YES`** | `NO_REANNOTATE_REQUIRED` | Retain clean pool image; re-annotate or remap `(0,3)<->(4,5)` / numeric labels |
| **`EXCLUDE_DESTRUCTIVE_AUGMENTATION`** | **2,146** (`9.10%`) | **2,146** (`23.02%`) | `NO` | `NO` | Exclude permanently (`1,698` synthetic noise-cutout SHAs + `448` 2×2 Mosaic SHAs) |
| **`HUMAN_REVIEW_REQUIRED`** | **1,813** (`7.69%`) | **1,813** (`19.44%`) | `CONDITIONAL` | `NO` | Human frame-by-frame triage (`882` `youtube-N` broadcast/pool mix + `931` `UNCERTAIN` crops) |
| **`EXCLUDE_OUT_OF_SCOPE`** | **1,094** (`4.64%`) | **1,094** (`11.73%`) | `NO` | `NO` | Exclude permanently (ocean, beach, river, lake, maritime scenes) |
| **`KEEP_IMAGE_REVIEW_LABEL`** | **737** (`3.12%`) | **737** (`7.90%`) | **`YES`** | `CONDITIONAL_PENDING_VISUAL_CHECK` | Retain pool image (`cls=5,6`); perform rapid human visual verification of boxes |
| **`EXCLUDE_WRONG_OBJECT`** | **53** (`0.22%`) | **53** (`0.57%`) | `NO` | `NO` | Exclude permanently (`SF05_visdrone_0000` aerial urban street pedestrians/vehicles) |
| **`KEEP_CANDIDATE`** | **18** (`0.08%`) | **18** (`0.19%`) | **`YES`** | **`YES_VERIFIED`** | Visually verified unique pool images with verified clean `5`/`6` semantic boxes |
| **Total** | **23,586** (`100.00%`) | **9,324** (`100.00%`) | **`4,218` `YES`** | **`18` Verified + `737` Pending** | **Total Recoverable Unique Pool Images = `4,218` (`45.24%` of unique SHAs)** |

---

## 3. Recoverable Pool Candidates by Super-Family (`4,218` Unique `image_sha256` Images)

Below is the exact distribution of the **`4,218` recoverable unique `image_sha256` swimming-pool images** (`KEEP_CANDIDATE` + `KEEP_IMAGE_REVIEW_LABEL` + `KEEP_IMAGE_REANNOTATE`) across the Super-Families:

| Super-Family | Unique `SHA-256` Total | `KEEP_CANDIDATE` (Verified `5/6`) | `KEEP_IMAGE_REVIEW_LABEL` (Unverified `5/6`) | `KEEP_IMAGE_REANNOTATE` (Offset / Numeric) | Total Recoverable Unique Pool Images | Why Re-Annotation / Review Is Needed & Specific Caveats |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`SF08_drowningN_mp4`** | 1,352 | 0 | 0 | **1,111** | **1,111** | Wave-pool rescue (`drowning1/47/50..55`, `9` video keys of 1 incident) & `drowning20`/`2`; suffers from `(0,3)<->(4,5)` shift & patch rotations |
| **`SF02_output_NNNN`** | 945 | **2** | **39** | **697** | **738** | `output1..75` (`41` unique clean SHAs) has clean `Class 5 (drowning)`; `output_0001..0600` (`697` unique clean SHAs) is 1 single video with numeric `0,2,3` |
| **`SF04_pure_digits`** | 1,197 | **15** | **572** | **87** | **674** | `1..200_jpg` (`587` unique clean SHAs) has `Class 6 (swimming)` (check for box-patch rotation); `130000_jpg` (`87` SHAs) has numeric `1;3` |
| **`SF03_imageNNNN`** | 734 | 0 | 0 | **552** | **552** | All `552` unique clean SHAs are **1 single Thai TV news CCTV video (`THAIRATH TV 32`)** with burned-in TV banners/anchor and incomplete `cls=4` labels |
| **`SF10_video_2022_12_07`** | 501 | 0 | 0 | **386** | **386** | 4 staged indoor tiled pool videos (`23-08-07/12/15/17_mp4`); clean pool pixels but `0<->4` class offset |
| **`SF12_clipchamp_mp4`** | 247 | **1** | **40** | **158** | **199** | `Untitled-video...` (`158` SHAs, Helsinki pool CCTV with `(0,3)<->(4,5)` shift); `-Clipchamp-_mp4` (`41` clean SHAs after dropping `50` cutout SHAs) |
| **`SF13_stock_named`** | 645 | 0 | 0 | **192** | **192** | Pool stock videos/photos (`child-pretending...`, `the-child-is-drowning...`, `istockphoto-1001608692`, etc.); `0<->4` class offset & watermarks |
| **`SF06_kakaotalk`** | 231 | 0 | 0 | **175** | **175** | 8 Korean swimming-pool batches; clean pool scenes with numeric classes `0, 2, 3` (`2` = floating children, `3` = active swimmers) |
| **`SF18_other_static`** | 1,045 | 0 | **86** | **69** | **155** | Static web/stock pool photos (`86` with `cls=5/6`, `69` with `0<->4` offset) |
| **`SF15_img_pic_photo`** | 127 | 0 | 0 | **36** | **36** | Underwater/poolside stock photos (`images-1, 3, 4, 14, 19, 24`) with `0<->4` class offset |
| **Other 7 Families** | 2,300 | 0 | 0 | 0 | **0** | `SF01_youtube_N` (`882` in `HUMAN_REVIEW_REQUIRED`), `SF05/07/11/14/17` (`OUT_OF_SCOPE`), `SF09` (`UNCERTAIN`) |
| **Total** | **9,324** | **18** | **737** | **3,463** | **4,218** | **`4,218` unique `image_sha256` pool images (`1,935` after capping giant single videos)** |

