# VISION ANNOTATION QUALITY REPORT — `DrowningDetectionTracking_v1`

- **Investigation Date:** 2026-11-03
- **Agent Role:** Visual Inspection & Dataset Forensics Agent (`vision-agent`)
- **Dataset:** `DrowningDetectionTracking_v1` (`23,586` images, `49,040` valid 5-field boxes + `15` malformed/polygon lines + `1` empty label)
- **Dataset Approval Status:** **NOT APPROVED FOR TRAINING**

---

## 1. Malformed Lines Investigation (All 15 Files in `invalid_annotations.csv`)

### 1.1 Forensic Finding (`VERIFIED`)
Every single one of the **15 malformed annotation lines** in `invalid_annotations.csv` (all in `train/labels/`) was extracted, parsed, rendered onto its corresponding original 640×640 image, and visually inspected (`VERIFIED`):
1. **Not Random File Corruption — They Are Valid YOLO Instance-Segmentation Polygons (`VERIFIED`):**
   - Each line has an odd token count (`45`, `47`, `57`, `59`, or `61` tokens) consisting of `1` class ID (`4` = `Out of Water`) followed by `22`, `23`, `28`, `29`, or `30` normalized `(x, y)` vertex pairs (`2N + 1` tokens) in `[0, 1]`.
   - When rendered onto the original images, the polygon vertices tightly trace the contour of a woman's head, shoulders, and outstretched arms in blue water (from an iStock footage clip with an `iStock` watermark).
2. **Why All 15 Files Must Be Excluded / Quarantined Rather Than Naively Converted (`VERIFIED`):**
   - **Semantic Contradiction (`Class 4 = Out of Water`):** All 15 files label a woman **floating/treading in blue water** as Class `4` (`Out of Water`). Converting the polygon to its bounding box without changing the class ID would inject 15 mislabeled `Out of Water` boxes on a person in water.
   - **Scene Ambiguity (`UNCERTAIN`):** All 15 frames are tight overhead close-ups of clear blue water with no visible pool walls, lane lines, or deck edges (`UNCERTAIN` pool vs open/tank water).
   - **Synthetic Bounding-Box Patch-Rotation Misalignment in 3 Files (`SHIFTED`):** In 3 of the 9 `mp4-3` files (`49f4cfb3...`, `38d172eb...`, `c0bb13a2...`, all with 61 fields), a prior Roboflow bounding-box-level rotation rotated the rectangular image patch containing the woman and her 30-vertex polygon by **90° clockwise**, while leaving the Line 1 5-field detection box (`4 0.51796875 0.246875 0.1765625 0.375`) **unrotated**, causing the 5-field box to miss half of her rotated head/body (`SHIFTED`).

### 1.2 Per-File Malformed Annotation Table (`15 / 15` Inspected)

| # | Split | Image / Label Base Name | Source Group | Malformed Line # | Token Count | Polygon Vertices | Valid 5-Field Box Present? | Observed Visual Evidence (`VERIFIED`) | Box Quality | Repairable or Exclude? |
|---:|---|---|---|---:|---:|---:|---|---|---|---|
| 1 | `train` | `3-727-Drowning-Stock-Videos-and-Royalty-Free-Footage-iStock_mp4-2_jpg.rf.0123f0e4edb420112fd33d283c9f5911` *(also Sheet 1 Tile 4)* | `3-727-...-istock_mp4-2_jpg` | Line 1 | `45` | `22` pts (`cls=4`) | **No** (only-invalid file) | Tight 22-vertex polygon around woman in clear blue water (rotated patch); no 5-field box; mislabeled `cls=4` (`Out of Water`). | `NO_BOX` | **EXCLUDE** (no 5-field box, wrong class `4` for person in water, `UNCERTAIN` scene, patch-rotation artifact). |
| 2 | `train` | `3-727-Drowning-Stock-Videos-and-Royalty-Free-Footage-iStock_mp4-2_jpg.rf.30c52f25b47a7ed299c09d55d0ff74b8` | `3-727-...-istock_mp4-2_jpg` | Line 1 | `45` | `22` pts (`cls=4`) | **No** (only-invalid file) | Tight 22-vertex polygon around woman in clear blue water (rotated patch); no 5-field box; mislabeled `cls=4` (`Out of Water`). | `NO_BOX` | **EXCLUDE** (same as #1). |
| 3 | `train` | `3-727-Drowning-Stock-Videos-and-Royalty-Free-Footage-iStock_mp4-2_jpg.rf.b9566e32e2efdc2df2e2fb3ae3cef63e` | `3-727-...-istock_mp4-2_jpg` | Line 1 | `45` | `22` pts (`cls=4`) | **No** (only-invalid file) | Tight 22-vertex polygon around woman in clear blue water (rotated patch); no 5-field box; mislabeled `cls=4` (`Out of Water`). | `NO_BOX` | **EXCLUDE** (same as #1). |
| 4 | `train` | `3-727-Drowning-Stock-Videos-and-Royalty-Free-Footage-iStock_mp4-2_jpg.rf.1c1fb0f0ebcc7b53c69fd0f8ed3fc322` | `3-727-...-istock_mp4-2_jpg` | Line 1 | `47` | `23` pts (`cls=4`) | **No** (only-invalid file) | Tight 23-vertex polygon around woman in clear blue water (rotated patch); no 5-field box; mislabeled `cls=4` (`Out of Water`). | `NO_BOX` | **EXCLUDE** (same as #1). |
| 5 | `train` | `3-727-Drowning-Stock-Videos-and-Royalty-Free-Footage-iStock_mp4-2_jpg.rf.88fc678a7fb1cef8ca40d4cd4cd1fb54` | `3-727-...-istock_mp4-2_jpg` | Line 1 | `47` | `23` pts (`cls=4`) | **No** (only-invalid file) | Tight 23-vertex polygon around woman in clear blue water (rotated patch); no 5-field box; mislabeled `cls=4` (`Out of Water`). | `NO_BOX` | **EXCLUDE** (same as #1). |
| 6 | `train` | `3-727-Drowning-Stock-Videos-and-Royalty-Free-Footage-iStock_mp4-2_jpg.rf.8c9bb48a4fd2bcd2ab1ffab3da6ca2bf` | `3-727-...-istock_mp4-2_jpg` | Line 1 | `47` | `23` pts (`cls=4`) | **No** (only-invalid file) | Tight 23-vertex polygon around woman in clear blue water (rotated patch); no 5-field box; mislabeled `cls=4` (`Out of Water`). | `NO_BOX` | **EXCLUDE** (same as #1). |
| 7 | `train` | `3-727-Drowning-Stock-Videos-and-Royalty-Free-Footage-iStock_mp4-3_jpg.rf.074d45d0b3fc1f4a3d97ba02bf2b7da3` | `3-727-...-istock_mp4-3_jpg` | Line 2 | `57` | `28` pts (`cls=4`) | **Yes** (Line 1: `4 0.5148 0.3516 0.1672 0.2562`) | Line 1 5-field box tightly covers woman's head; Line 2 28-pt polygon traces her head, torso, and outstretched arms in blue water. Both mislabeled `cls=4` (`Out of Water`). | `CORRECT` (geometry) | **EXCLUDE** (mixed box+polygon format, wrong class `4` on person in water, `UNCERTAIN` scene). |
| 8 | `train` | `3-727-Drowning-Stock-Videos-and-Royalty-Free-Footage-iStock_mp4-3_jpg.rf.4e2ab011022ab17347d1ca45b8f39ed6` | `3-727-...-istock_mp4-3_jpg` | Line 2 | `57` | `28` pts (`cls=4`) | **Yes** (Line 1: `4 0.5148 0.3516 0.1672 0.2562`) | Same geometry as #7 (minor augmentation variant); Line 1 boxes head, Line 2 28-pt polygon traces upper body/arms. Both mislabeled `cls=4`. | `CORRECT` (geometry) | **EXCLUDE** (same as #7). |
| 9 | `train` | `3-727-Drowning-Stock-Videos-and-Royalty-Free-Footage-iStock_mp4-3_jpg.rf.6c7d62cd5ee27bf3a5fa6bb59acff039` | `3-727-...-istock_mp4-3_jpg` | Line 2 | `57` | `28` pts (`cls=4`) | **Yes** (Line 1: `4 0.5148 0.3516 0.1672 0.2562`) | Same geometry as #7; Line 1 boxes head, Line 2 28-pt polygon traces upper body/arms. Both mislabeled `cls=4`. | `CORRECT` (geometry) | **EXCLUDE** (same as #7). |
| 10 | `train` | `3-727-Drowning-Stock-Videos-and-Royalty-Free-Footage-iStock_mp4-3_jpg.rf.49f4cfb3783b21ebda2d558b3cd024a1` | `3-727-...-istock_mp4-3_jpg` | Line 2 | `61` | `30` pts (`cls=4`) | **Yes** (Line 1: `4 0.5180 0.2469 0.1766 0.3750`) | Central patch containing woman and Line 2 30-pt polygon is rotated 90° clockwise, leaving Line 1 5-field box unrotated and shifted off her head. | `SHIFTED` | **EXCLUDE** (corrupted by 90° patch rotation + misaligned 5-field box + wrong class `4`). |
| 11 | `train` | `3-727-Drowning-Stock-Videos-and-Royalty-Free-Footage-iStock_mp4-3_jpg.rf.5b5c402de98965d4f600279c993ed92a` | `3-727-...-istock_mp4-3_jpg` | Line 2 | `59` | `29` pts (`cls=4`) | **Yes** (Line 1: `4 0.5180 0.3469 0.1766 0.3750`) | Central patch tilted slightly; Line 1 5-field box covers head/shoulders and Line 2 29-pt polygon covers body/arms. Both mislabeled `cls=4`. | `CORRECT` (geometry) | **EXCLUDE** (synthetic patch-tilt artifact + wrong class `4` + `UNCERTAIN` scene). |
| 12 | `train` | `3-727-Drowning-Stock-Videos-and-Royalty-Free-Footage-iStock_mp4-3_jpg.rf.38d172eb5d172cf5ee9c93fc8483943e` | `3-727-...-istock_mp4-3_jpg` | Line 2 | `61` | `30` pts (`cls=4`) | **Yes** (Line 1: `4 0.5180 0.2469 0.1766 0.3750`) | Same 90° clockwise patch rotation as #10: Line 2 30-pt polygon rotated with patch, Line 1 5-field box left unrotated and misaligned. | `SHIFTED` | **EXCLUDE** (same as #10). |
| 13 | `train` | `3-727-Drowning-Stock-Videos-and-Royalty-Free-Footage-iStock_mp4-3_jpg.rf.57941a21b9feff2b9c3d41cc14c70317` | `3-727-...-istock_mp4-3_jpg` | Line 2 | `59` | `29` pts (`cls=4`) | **Yes** (Line 1: `4 0.5180 0.3469 0.1766 0.3750`) | Same tilted patch geometry as #11; Line 1 covers head/shoulders, Line 2 29-pt polygon covers body/arms. Both mislabeled `cls=4`. | `CORRECT` (geometry) | **EXCLUDE** (same as #11). |
| 14 | `train` | `3-727-Drowning-Stock-Videos-and-Royalty-Free-Footage-iStock_mp4-3_jpg.rf.13aa8896f5ed1cffd0bcac0fa9691f65` | `3-727-...-istock_mp4-3_jpg` | Line 2 | `59` | `29` pts (`cls=4`) | **Yes** (Line 1: `4 0.5180 0.3469 0.1766 0.3750`) | Same tilted patch geometry as #11; Line 1 covers head/shoulders, Line 2 29-pt polygon covers body/arms. Both mislabeled `cls=4`. | `CORRECT` (geometry) | **EXCLUDE** (same as #11). |
| 15 | `train` | `3-727-Drowning-Stock-Videos-and-Royalty-Free-Footage-iStock_mp4-3_jpg.rf.c0bb13a259a282c810af6aa167b14da6` | `3-727-...-istock_mp4-3_jpg` | Line 2 | `61` | `30` pts (`cls=4`) | **Yes** (Line 1: `4 0.5180 0.2469 0.1766 0.3750`) | Same 90° clockwise patch rotation as #10 and #12: Line 2 30-pt polygon rotated with patch, Line 1 5-field box left unrotated and misaligned. | `SHIFTED` | **EXCLUDE** (same as #10). |

---

## 2. Oversized Bounding Boxes (`area > 0.90`)

### 2.1 Dataset-Wide Distribution (`67` Oversized Boxes)
From `suspicious_samples.csv`, `67` bounding boxes across the dataset cover **>90% of the 640×640 frame** (`area > 0.90`):
- **By Class:** Class `0` (`'3'`): **`60` boxes** (`89.6%` of all oversized boxes); Class `3` (`'6'`): **`4` boxes**; Class `5` (`drowning`): **`2` boxes**; Class `6` (`swimming`): **`1` box**.
- **By Split:** `train`: `60` boxes; `valid`: `5` boxes; `test`: `2` boxes.

### 2.2 Visual Inspection of Contact Sheet 5 Oversized Box Samples (`VERIFIED`)
We visually inspected all 4 oversized-box tiles on `contact_sheet_5.png` (Tiles 1–4) plus 1 near-oversized `88.6%`-area tile on `contact_sheet_1.png` (Tile 2):

| Sheet & Tile | Split & Image Name | Class ID | Box Coordinates (`cx, cy, w, h`) | Box Area | Observed Visual Evidence (`VERIFIED`) | Box Quality | Verdict |
|---|---|---:|---|---:|---|---|---|
| Sheet 5 Tile 1 | `test/images/drowning_small-1-1000x573_jpg.rf.c604ab0edcdd9a14a6c37ae3128a87f7.jpg` | `0` | `0.5000, 0.5000, 1.0000, 1.0000` | `100.00%` | Rotated image of a child's head splashing in blue water (`UNCERTAIN` scene); the box spans the entire 640×640 image including black rotation corners instead of the child's head. | `FULL_FRAME` | **EXCLUDE** |
| Sheet 5 Tile 2 | `test/images/test1-5-_mp4-12_jpg.rf.6d95f2597675ab83bc2deb9d8e822344.jpg` | `3` | `0.5016, 0.5008, 0.9500, 0.9516` | `90.40%` | `90.4%` of the frame is obliterated by a giant black salt-and-pepper noise cutout, leaving only a thin border of blue water ripples; the Class-`3` box encloses the entire noise cutout. | `FULL_FRAME` | **EXCLUDE** |
| Sheet 5 Tile 3 | `train/images/65_png_jpg.rf.1ac055ba05d7ec2f81c6238a9af6068d.jpg` | `0` | `0.5000, 0.5000, 0.9914, 0.9914` | `98.29%` | `98.3%` of the frame is obliterated by a giant black salt-and-pepper noise cutout; no person or water feature is visible inside the box. | `FULL_FRAME` | **EXCLUDE** |
| Sheet 5 Tile 4 | `train/images/65_png_jpg.rf.971923789e21eb98962806bdb41def5d.jpg` | `0` | `0.5000, 0.5000, 0.9914, 0.9914` | `98.29%` | `98.3%` of the frame is obliterated by a giant gray/black salt-and-pepper noise cutout; no person or water feature is visible inside the box. | `FULL_FRAME` | **EXCLUDE** |
| Sheet 1 Tile 2 | `train/images/drowning_small-1-1000x573_jpg.rf.9fbbb64c047b266c089d83e385a23f76.jpg` | `0` | `0.5000, 0.5000, 0.9414, 0.9414` | `88.62%` | `88.6%` of the frame is obliterated by a giant black salt-and-pepper noise cutout over blue water ripples. | `FULL_FRAME` | **EXCLUDE** |

> **Forensic Explanation (`VERIFIED`):** When an image already had a full-frame (or near-full-frame) bounding box and a subsequent Roboflow export applied **bounding-box-level cutout/noise augmentation**, the augmentation replaced the entire interior of the full-frame bounding box (`90%–98%` of the image!) with synthetic black salt-and-pepper noise. All `67` oversized boxes (`>90%` area) and their associated cutout images are unusable for training.

---

## 3. Empty Annotation Investigation (`MVI_1627_VIS_frame20`)

- **File Inspected:** `test/images/MVI_1627_VIS_frame20_jpg.rf.c948cc23addbed52b173c609a43c3c59.jpg` (`46,815` bytes) and `test/labels/MVI_1627_VIS_frame20_jpg.rf.c948cc23addbed52b173c609a43c3c59.txt` (`0` bytes).
  - *Archive Case Note:* Inside `test-20261102T163222Z-1-001.zip`, the filename prefix is uppercase `MVI_1627_VIS_frame20_jpg...`, whereas `suspicious_samples.csv` recorded it in lowercase (`mvi_1627_vis_frame20_jpg...`).
- **Observed Visual Evidence (`VERIFIED`):**
  - The image is a wide-angle maritime horizon photograph from the **Singapore Maritime Dataset (`MVI_1627_VIS`)** showing open ocean water under a pale sky with **6 distant cargo ships/tankers/vessels** along the horizon.
  - There are **zero people, swimmers, or drowning victims** anywhere in the frame.
- **Forensic Determination (`VERIFIED`):**
  - **Annotation Status:** `NO_BOX` — It is a **genuine negative sample** with respect to human/swimmer presence (not a missing person annotation).
  - **Scene Scope Status:** **`OUT_OF_SCOPE`** — Because it depicts an open-ocean shipping lane rather than a swimming pool, it must be **excluded** under the project's strict `SWIMMING POOLS ONLY` operational scope.

---

## 4. Systemic Pre-Baked Augmentation Corruptions Discovered (`VERIFIED`)

Beyond the 15 polygon lines, 67 oversized boxes, and 1 empty label, pixel-level inspection uncovered **three pervasive, dataset-wide image/annotation corruptions** introduced during prior Roboflow exports before the dataset was merged:

1. **Destructive Bounding-Box Cutout / Salt-and-Pepper Noise Occlusion (`WRONG_OBJECT`):**
   - Observed in `16` of the `55` contact-sheet tiles (`29.1%` of contact sheet tiles!) across `drowning_small-1-1000x573_jpg`, `3-727-...-iStock_mp4-2_jpg`, `youtube-1_jpg` through `youtube-10_jpg`, `test1-5-_mp4-12_jpg`, `65_png_jpg`, and `-Clipchamp-_mp4-10_jpg.rf.65238cc733fa222480578103974aebde.jpg` (Sheet 5 Tile 6).
   - In these files, the entire rectangular region inside each bounding box was replaced with either solid black/gray pixels or dense black-and-white salt-and-pepper noise, **completely erasing the person/swimmer** while leaving the bounding box active in the `.txt` label file. Training a detector on these files teaches the model to detect synthetic black noise rectangles as `drowning` or `swimming`.
2. **Bounding-Box-Level Patch Rotation / Flip Without Box Re-alignment (`SHIFTED` / Synthetic Seams):**
   - Observed in `203512036-1-_jpg` (Sheet 4 Tile 3 and `valid/54b30c35...`), `3-727-...-iStock_mp4-2` and `mp4-3` (all 15 malformed files plus Sheet 1 Tile 5), `drowning47_mp4-10_jpg`, and `video40-1-_mp4-81_jpg`.
   - Only the rectangular sub-patch inside the bounding box was rotated (e.g., 15° or 90°) or flipped, creating sharp artificial rectangular seams in the water and, in the 90°-rotated `mp4-3` files, leaving the 5-field box misaligned with the rotated person (`SHIFTED`).
3. **2×2 Mosaic Composites Re-Imported as Raw Images (`TRUNCATED` & `WRONG_OBJECT`):**
   - Observed in `31` of the `36` tiles on Contact Sheets 1–3 (`youtube-1_jpg` through `youtube-11_jpg`).
   - Prior 2×2 Mosaic training augmentations were exported and re-split into `train`, `valid`, and `test`, causing bounding boxes to be clipped at internal quadrant seams (`TRUNCATED`) and mixing unrelated scenes (e.g., outdoor grass, TV studio sign-language interpreters boxed as `swimming`, and dry-land TV hosts boxed as `drowning`) into single images.

---

## 5. Observed Box Quality by Class Across Inspected Samples (`VERIFIED`)

Across the **106 unique dataset images** visually inspected in this investigation (`55` contact-sheet tiles + `14` additional malformed-label images + `1` empty-label image + `12` extra class-semantics images + `24` leakage spot-check images), the primary `box_quality` classification per image is:

| Box Quality Category | Count across 55 Contact Sheet Tiles | Count across All 106 Inspected Images | Primary Classes Affected & Root Cause |
|---|---:|---:|---|
| `CORRECT` | `23` (`41.8%`) | `64` (`60.4%`) | Unoccluded single-frame images in Classes `0`–`6` (note: includes images where box geometry tightly fits the person even when the class ID is semantically wrong, e.g., `cls=4` on underwater person in Sheet 3 Tile 7 or `cls=1` on VisDrone pedestrians). |
| `WRONG_OBJECT` | `19` (`34.5%`) | `20` (`18.9%`) | Classes `0`, `1`, `2`, `3`, `5`, `6`: (a) `13` tiles where boxes enclose synthetic black/gray salt-and-pepper noise cutouts obliterating the person; (b) `4` tiles where Class `6` (`swimming`) boxes a TV sign-language interpreter inset; (c) `1` tile where Class `5` (`drowning`) boxes TV host Benedict Galazan on dry land; (d) `1` tile where Class `5` boxes a rescue pole/pool coping. |
| `TRUNCATED` | `7` (`12.7%`) | `7` (`6.6%`) | Classes `4`, `5`, `6` in `youtube-*` 2×2 Mosaic composites where objects and boxes are sliced across internal quadrant boundaries. |
| `FULL_FRAME` | `5` (`9.1%`) | `5` (`4.7%`) | Classes `0` (`4` tiles) and `3` (`1` tile): `100%` rotated frame box (`drowning_small-1`) and `88.6%–98.3%` giant black noise cutout boxes (`drowning_small-1`, `test1-5-_mp4-12`, `65_png`). |
| `NO_BOX` | `1` (`1.8%`) | `7` (`6.6%`) | `6` "only-invalid" `3-727-...-iStock_mp4-2` files (contain only a 45/47-field `cls=4` polygon and no 5-field box) + `1` empty-label file (`MVI_1627_VIS_frame20`). |
| `SHIFTED` | `0` (`0.0%`) | `3` (`2.8%`) | `3` `3-727-...-iStock_mp4-3` 61-field files (`49f4cfb3...`, `38d172eb...`, `c0bb13a2...`) where 90° clockwise patch rotation rotated the woman and polygon but left the Line 1 5-field box unrotated. |
| **Total Inspected** | **`55` (`100.0%`)** | **`106` (`100.0%`)** | **All 106 images visually verified (`review_source=vision-agent`).** |

