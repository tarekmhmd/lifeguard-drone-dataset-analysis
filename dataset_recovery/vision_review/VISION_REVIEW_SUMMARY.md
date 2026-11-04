# VISION REVIEW SUMMARY — `DrowningDetectionTracking_v1`

- **Investigation Date:** 2026-11-03
- **Agent Role:** Visual Inspection & Dataset Forensics Agent (`vision-agent`)
- **Dataset:** `DrowningDetectionTracking_v1` (`23,586` images, `49,040` valid 5-field boxes + `15` segmentation polygon lines + `1` empty label across `train`/`valid`/`test`)
- **Strict Operational Scope:** Autonomous AI Lifeguard System — **SWIMMING POOLS ONLY**
- **Dataset Readiness Status:** **NOT APPROVED FOR TRAINING**

---

## 1. Investigation Coverage: Expected vs. Actual Targets Visually Inspected

Every required target defined in `03_source_investigation/VISION_AGENT_TASK_PACKAGE.md` was extracted from the read-only dataset archives into a temporary scratch workspace and visually inspected at pixel level (`106` unique dataset images + `5` contact sheets + `27` rendered overlay images = **`138` image files visually inspected**):

| Investigation Batch / Task Area | Expected Targets in Task Package | Actual Targets Visually Inspected (`review_source=vision-agent`) | Completion Status |
|---|---:|---:|---|
| **1. Scene & Group Homogeneity (`contact_sheet_1.png`–`contact_sheet_3.png`)** | `36` tiles | **`36 / 36` tiles** (both on contact sheets and as full-resolution 640×640 original images) | **100% Complete (`VERIFIED`)** |
| **2. Numeric Class Semantics (`contact_sheet_4.png` + extra independent groups)** | `12` tiles on Sheet 4 (+ independent source group checks for Classes `0`–`3`) | **`12 / 12` Sheet 4 tiles** + **`12` additional images** across independent source groups (`24` images total) | **100% Complete (`VERIFIED`)** |
| **3. Oversized Boxes & Prior Classifications (`contact_sheet_5.png`)** | `7` tiles (`4` oversized boxes + `3` `-Clipchamp-_mp4-10` verification tiles) | **`7 / 7` tiles** (full-resolution 640×640 original images + box coordinates verified) | **100% Complete (`VERIFIED`)** |
| **4. Malformed Annotations (`invalid_annotations.csv`)** | `15` entries in `train/labels/` | **`15 / 15` files** (`1` on Sheet 1 Tile 4 + `14` additional files; both raw 640×640 images and `15` rendered box+polygon overlays inspected) | **100% Complete (`VERIFIED`)** |
| **5. Empty Annotation (`mvi_1627_vis_frame20`)** | `1` empty label file in `test/labels/` | **`1 / 1` file** (`test/images/MVI_1627_VIS_frame20_jpg.rf.c948cc23addbed52b173c609a43c3c59.jpg`) | **100% Complete (`VERIFIED`)** |
| **6. Duplicate Detection & Cross-Split Leakage Spot-Checks** | Byte-hash (`SHA-256`) & cross-split source-prefix spot-checks | **`24` additional images** across `5` cross-split exact-image-hash pairs (`10` images), `3` cross-split exact-label-hash pairs (`6` images), and `3` cross-split source prefixes (`8` images, plus `203512036-1` Tile 39, `drowning_small-1`, and `-Clipchamp-_mp4-10`) | **100% Complete (`VERIFIED`)** |
| **Total Unique Dataset Images Recorded in `VISION_REVIEW_RESULTS.csv`** | `70+` minimum target images | **`106` unique dataset images (`106` rows in `VISION_REVIEW_RESULTS.csv`)** | **100% Complete (`VERIFIED`)** |

---

## 2. Scene Classification Totals (`POOL` vs. `OUT_OF_SCOPE` vs. `UNCERTAIN`)

### 2.1 Breakdown Across Inspected Images (`VERIFIED`)

| Scene Category | Contact Sheets 1–3 (`36` Homogeneity Tiles) | Contact Sheet 4 (`12` Class-Semantics Tiles) | Contact Sheet 5 (`7` Oversized & Audit Tiles) | All `55` Contact Sheet Tiles | All `106` Visually Inspected Dataset Images |
|---|---:|---:|---:|---:|---:|
| **`POOL`** | `6` (`16.7%`) | `4` (`33.3%`) | `3` (`42.9%`) | **`13` (`23.6%`)** | **`33` (`31.1%`)** |
| **`OUT_OF_SCOPE`** | `0` (`0.0%`) | `7` (`58.3%`) | `0` (`0.0%`) | **`7` (`12.7%`)** | **`18` (`17.0%`)** |
| **`UNCERTAIN`** | `30` (`83.3%`) | `1` (`8.3%`) | `4` (`57.1%`) | **`35` (`63.6%`)** | **`55` (`51.9%`)** |
| **Total** | **`36`** | **`12`** | **`7`** | **`55` (`100.0%`)** | **`106` (`100.0%`)** |

### 2.2 Key Scene Findings (`VERIFIED`)
1. **Why `UNCERTAIN` Dominates Contact Sheets 1–3 (`30 / 36` tiles):**
   - `31` of the `36` tiles on Contact Sheets 1–3 are **2×2 Roboflow Mosaic composites** (`youtube-1_jpg` through `youtube-11_jpg`) that stitch together 4 unrelated video frames (mixing swimming pools, close-up water with no visible pool walls, TV news studio sign-language insets, and outdoor dry grass/plaza scenes) or close-up water crops (`drowning_small-1-1000x573_jpg`, `3-727-...-iStock_mp4-2_jpg`) where no pool boundary is visible.
   - Only `6` tiles on Sheets 1–3 (`Sheet 1 Tile 12`, `Sheet 2 Tiles 6 & 12`, `Sheet 3 Tiles 3, 7, & 12`) depict single-frame swimming pools or 4-quadrant all-pool mosaics (`POOL`), and `4` of those `6` have their swimmers obliterated by black salt-and-pepper noise cutouts!
2. **Confirmed `OUT_OF_SCOPE` Contamination (`18` Inspected Images across Multiple Subsets):**
   - **VisDrone Aerial Urban Street/Traffic Subset (`0000001_*`, `0000026_*`, `0000103_*`, `0000116_*`, `0000242_*`):** Aerial drone views of dry urban streets, taxis, intersections, and pedestrians (`6` images inspected, all `OUT_OF_SCOPE`).
   - **Singapore Maritime Dataset (`MVI_1627_VIS_frame20`):** Open-ocean horizon with 6 cargo vessels (`1` image inspected, `OUT_OF_SCOPE`).
   - **Open-Water / Sea / River Subsets (`10_MOV-52_jpg`, `124_png_jpg`, `129_png_jpg`, `134_png_jpg`, `10687_jpg`, `11159_jpg`, `11776_jpg`, `val5_png_jpg`, `video40-1-_mp4-81_jpg`):** Murky rivers, dark open-water waves, aerial green sea views, sunset ocean waves, and ocean surf (`11` images inspected, all `OUT_OF_SCOPE`).
3. **Correction of Prior `-Clipchamp-_mp4-10_jpg` Conflict on Contact Sheet 5 (Tiles 5–7):**
   - Prior records listed `2b6633cd...` as `POOL`, `65238cc7...` as `OUT_OF_SCOPE`, and `a355686f...` as `UNCERTAIN`.
   - Direct pixel inspection (`VERIFIED`) confirmed that **all three images are augmented versions of the exact same swimming pool frame** showing a `"5 FT"` depth marker on the pool wall and two swimmers with pool noodles (`POOL`). However, `65238cc7...` (Tile 6) has both swimmers replaced by synthetic black salt-and-pepper noise cutouts (`box_quality = WRONG_OBJECT`).

---

## 3. Class-Semantics Confidence Summary (Classes 0–6)

| Class ID | YAML Name | Total Valid Boxes | Independent Source Groups Inspected | Semantic Confidence | Summary of Visual Evidence (`VERIFIED`) |
|---:|---|---:|---:|---|---|
| **`0`** | `'3'` | `8,979` | `11` | **`UNKNOWN`** | Mixes submerged/drowning people (`10_MOV-52`, `124_png`, `129_png`, `134_png`, `image1944`, `youtube-10`), non-drowning child in lifejacket (`203512036-1`), synthetic black/gray noise cutouts (`drowning_small`, `3-727-...-iStock`, `youtube-*`), and `60` full-frame (`>90%` area) boxes (`drowning_small`, `65_png`). |
| **`1`** | `'4'` | `2,994` | `8` | **`UNKNOWN`** | **Fatal Schema Collision:** Boxes **urban street pedestrians and motorbike riders** in VisDrone traffic scenes (`0000*`), **poolside/deck people** in `image1944`, `image724`, and `youtube-0`, and black noise cutouts in `youtube-10`. |
| **`2`** | `'5'` | `3,389` | `7` | **`UNKNOWN`** | **Fatal Schema Collision:** Boxes children/instructors **floating or swimming IN pool water** in `KakaoTalk_20230504_000756264_*` (`04`, `10`, `16`), but boxes adults/children **on the DRY pool deck** in `image1944`, `image724`, and `output_0001`. |
| **`3`** | `'6'` | `9,160` | `10` | **`UNKNOWN`** | Boxes active swimmers/rescuers in `10687`, `11159`, `11776`, `130163`, `130315`, and `youtube-10`, but in `youtube-1`, `youtube-3`, `youtube-7`, `youtube-8`, `youtube-9`, and `test1-5-_mp4-12` boxes **synthetic black/white salt-and-pepper noise cutouts** (including `4` oversized `>90%` area boxes). |
| **`4`** | `'Out of Water'` | `3,200` (+`15` polygons) | `8` | **`PARTIALLY VERIFIED`** | Boxes poolside/deck people in clean frames, but **mislabels people in/under water** as `Out of Water` in `youtube-11_jpg.rf.00fe6568...` (person completely underwater in pool), `drowning_small-1-1000x573_jpg.rf.021f153a...`, and all `15` `3-727-...-iStock` polygon files. |
| **`5`** | `'drowning'` | `11,226` | `11` | **`PARTIALLY VERIFIED`** | Boxes submerged/distressed people in water in clean frames, but also boxes upright TV host Benedict Galazan on dry land (`youtube-11_jpg.rf.86eddc88...`), a rescue pole/pool coping (`youtube-2_jpg.rf.043164e2...`), and mosaic-truncated fragments. |
| **`6`** | `'swimming'` | `10,092` | `10` | **`PARTIALLY VERIFIED`** | Boxes active swimmers in clean frames, but also boxes **TV broadcast sign-language interpreters** (`youtube-1`, `youtube-6`, `youtube-7`, `youtube-8`), black noise cutouts (`-Clipchamp-_mp4-10_jpg.rf.65238cc7...`), and mosaic-truncated fragments. |

---

## 4. Box-Quality Counts Summary (`VERIFIED`)

| `box_quality` Value | Count (`55` Contact Sheet Tiles) | Count (`106` Total Inspected Images) | Primary Visual Evidence |
|---|---:|---:|---|
| **`CORRECT`** | `23` (`41.8%`) | `64` (`60.4%`) | Tightly aligned boxes around visible people in single-frame images (or unclipped mosaic quadrants). *Note: includes cases where box geometry is tight on the person but the class ID or scene is invalid.* |
| **`WRONG_OBJECT`** | `19` (`34.5%`) | `20` (`18.9%`) | `14` images where boxes enclose synthetic black/gray salt-and-pepper noise cutouts obliterating the person; `4` images where Class `6` (`swimming`) boxes a TV sign-language interpreter inset; `1` image where Class `5` (`drowning`) boxes a TV host on dry land; `1` image where Class `5` boxes a rescue pole/coping edge. |
| **`TRUNCATED`** | `7` (`12.7%`) | `7` (`6.6%`) | 2×2 Mosaic composite images (`youtube-*`) where people and boxes are cut across internal quadrant boundaries. |
| **`FULL_FRAME`** | `5` (`9.1%`) | `5` (`4.7%`) | `100.0%` whole-image box (`drowning_small-1` Sheet 5 Tile 1) and `88.6%–98.3%` giant black noise cutout boxes (`drowning_small-1` Sheet 1 Tile 2, `test1-5-_mp4-12` Sheet 5 Tile 2, `65_png` Sheet 5 Tiles 3–4). |
| **`NO_BOX`** | `1` (`1.8%`) | `7` (`6.6%`) | `6` `3-727-...-iStock_mp4-2` files containing only a 45/47-field segmentation polygon and no 5-field detection box + `1` empty label file (`MVI_1627_VIS_frame20`). |
| **`SHIFTED`** | `0` (`0.0%`) | `3` (`2.8%`) | `3` `3-727-...-iStock_mp4-3` 61-field files (`49f4cfb3...`, `38d172eb...`, `c0bb13a2...`) where 90° clockwise patch rotation rotated the woman and polygon but left the Line 1 5-field box unrotated. |
| **Total** | **`55` (`100.0%`)** | **`106` (`100.0%`)** | **All `106` rows recorded with `review_source=vision-agent` and `review_status=REVIEWED`.** |

---

## 5. Duplicate and Leakage Classifications Summary (`VERIFIED`)

1. **`CONFIRMED_EXACT_DUP`:**
   - All `99` `exact_image_hash` groups (`365` files) in `duplicate_images.csv` are **cross-split (`train` vs `valid`) byte-identical duplicates**.
   - Visually verified across `5` cross-split pairs (`10` images) in `youtube-2_jpg`, `youtube-3_jpg`, `youtube-4_jpg`, `youtube-5_jpg`, and `youtube-6_jpg` with matching 64-hex `SHA-256` digests and byte sizes.
2. **`CONFIRMED_SAME_VIDEO`:**
   - All `173` `exact_label_hash` groups (`702` files) in `duplicate_images.csv` are **cross-split (`train`/`valid`/`test`) near-identical duplicates** (visually verified across `3` pairs / `6` images in `video_2022-12-07_23-08-07_mp4-2_jpg`, `video_2022-12-07_23-08-12_mp4-16_jpg`, and `val5_png_jpg`).
   - `1,616` of `4,949` source groups (`32.7%`) span multiple splits (`same_source_prefix.csv`); visual inspection of `5` multi-split prefixes (`203512036-1-_jpg`, `drowning47_mp4-10_jpg`, `video40-1-_mp4-81_jpg`, `drowning_small-1-1000x573_jpg`, `-Clipchamp-_mp4-10_jpg`) confirmed `100%` same-frame / same-video leakage across `train`, `valid`, and `test`.
   - Visual inspection also proved **cross-prefix video leakage**: `youtube-2_jpg`, `youtube-5_jpg`, `youtube-6_jpg`, and `youtube-10_jpg` share frames from the exact same indoor green-coping pool rescue video; `youtube-3_jpg` and `youtube-4_jpg` share frames from the exact same outdoor red-coping backyard pool video; `image724_jpg` and `image1944_jpg` share frames from the exact same Thai TV news CCTV broadcast.
3. **`SUSPECTED_SOURCE`:**
   - All `youtube-*` source groups (`>2,800` images) are **composite mixtures** containing 2×2 Mosaic images (`31 / 36` tiles on Sheets 1–3) that combine 4 unrelated videos per image and mix numeric (`0`–`3`) and semantic (`4`–`6`) schemas.
4. **`UNRESOLVED`:**
   - The remaining uninspected cross-split source groups in `same_source_prefix.csv` and sequential frame-number prefixes (`10687_jpg`, `11159_jpg`, `image724_jpg`, etc.) require dataset-wide perceptual hashing / embedding clustering before any leakage-free split can be constructed.

---

## 6. Remaining `PENDING_REVIEW` Items & Investigation Limitations

1. **Uninspected Dataset Remainder (`PENDING_REVIEW`):**
   - Out of `23,586` total images and `4,949` source groups in `DrowningDetectionTracking_v1`, this investigation visually inspected **`106` unique images across `38` distinct source groups** (`100%` of all required targets in `VISION_AGENT_TASK_PACKAGE.md`).
   - The remaining **`23,480` images** (`99.55%` of images) and **`4,911` source groups** (including the `4,877` `PENDING_REVIEW` source groups tracked in `SCENE_CLASSIFICATION_PROGRESS.csv` and the `62` remaining uninspected oversized boxes in `suspicious_samples.csv`) remain `PENDING_REVIEW`.
2. **Why `youtube-*` Groups Cannot Be Cleared by Prefix (`VERIFIED`):**
   - Because `youtube-*` groups mix 2×2 mosaics, multiple videos, TV studio overlays, and black noise cutouts within the same prefix, prefix-level review cannot salvage them; all 2×2 mosaic images and synthetic noise-cutout images must be filtered at the individual image level.

---

## 7. Overall Dataset Readiness Statement

> **FINAL READINESS VERDICT: `DrowningDetectionTracking_v1` IS NOT APPROVED FOR TRAINING.**
>
> Based on confirmed visual evidence (`VERIFIED`), `DrowningDetectionTracking_v1` suffers from:
> 1. **Unresolvable numeric class collisions (`Classes 0–3` = `50.0%` of all boxes)**, including VisDrone urban street pedestrians in Class `1` and contradictory in-water vs. dry-deck annotations in Class `2`;
> 2. **Widespread pre-baked destructive Roboflow augmentations**, including synthetic black/gray salt-and-pepper noise cutouts obliterating objects inside active bounding boxes, `67` full-frame (`>90%` area) boxes, 90° patch-rotation box misalignments, and re-imported 2×2 Mosaic composites;
> 3. **Severe semantic mislabeling in Classes `4`–`6`**, including underwater and in-water people labeled `Out of Water` (`Class 4`), dry-land TV hosts labeled `drowning` (`Class 5`), and TV sign-language interpreters labeled `swimming` (`Class 6`);
> 4. **Out-of-scope scene contamination** (VisDrone urban streets, Singapore Maritime cargo ships, open ocean surf, and rivers) violating the **SWIMMING POOLS ONLY** operational scope; and
> 5. **Systemic cross-split data leakage** (`99` byte-identical cross-split image groups, `173` cross-split identical label groups, `1,616` cross-split source prefixes, and cross-prefix shared videos).
>
> **The dataset remains strictly UNAPPROVED FOR TRAINING unless and until a comprehensive cleaning, filtering, deduplication, and re-splitting pipeline is executed and separate human approval is explicitly granted.**

