# VISION AGENT TASK PACKAGE — DrowningDetectionTracking_v1

This document is the single source of instructions for a separate **multimodal vision-capable agent** that can open and inspect local image files. It is self-contained: the agent does not need access to any prior conversation.

Project: **Autonomous AI Lifeguard System — swimming pools only.**

---

## A. Project background and scope

- The system monitors **swimming pools**, detects people, tracks them, and classifies behavior over time to distinguish normal swimming/playing/diving/floating from possible drowning/distress.
- **Detecting a person is not drowning detection.** Drowning is a temporal event; a single image or a person box cannot establish drowning.
- **Scope is pools only.** Beach, open-water, street, and drone scenes are out of scope for this dataset's role.
- `DrowningDetectionTracking_v1` is **one** dataset (still images, YOLO boxes). It is a candidate supplementary detection/behavior-image dataset. It is **NOT approved for training** and has unresolved class-schema and leakage problems.

## B. Dataset structure and statistics

Original dataset (READ-ONLY):

`E:\lifeguardDron\DrowningDetectionTracking_v1\`

| Archive | Contents |
|---|---|
| `train-20261102T163226Z-1-001.zip` | `train/images/*.jpg`, `train/labels/*.txt` |
| `valid-20261102T163228Z-1-001.zip` | `valid/images/*.jpg`, `valid/labels/*.txt` |
| `test-20261102T163222Z-1-001.zip` | `test/images/*.jpg`, `test/labels/*.txt` |

Metadata files in the same folder: `README.roboflow.txt`, `README.dataset.txt`, `data.fixed.yaml`. The original `data.yaml` is **missing**.

Statistics (VERIFIED by full read-only audit):

| Split | Images | Labels | Valid boxes |
|---|---:|---:|---:|
| train | 21,369 | 21,369 | 44,834 |
| valid | 1,215 | 1,215 | 2,655 |
| test | 1,002 | 1,002 | 1,551 |
| **Total** | **23,586** | **23,586** | **49,040** |

- All images are 640×640, decode successfully, no zero-byte or unsupported files.
- Image-label pairing is complete: 0 missing, 0 orphan, 0 duplicate stems.
- No invalid class IDs, no invalid coordinates, no out-of-bounds boxes, no duplicate annotation lines, no tiny boxes were found.

## C. Complete class inventory and unresolved class semantics

YOLO labels use `class_id x_center y_center width height` (normalized).

Class names (from `data.fixed.yaml`, `nc: 7`):

| ID | Name | Total boxes | Images containing class | Semantic status |
|---:|---|---:|---:|---|
| 0 | `3` | 9,890 | 8,597 | **UNKNOWN** (numeric name) |
| 1 | `4` | 4,683 | 370 | **UNKNOWN** (numeric name) |
| 2 | `5` | 1,432 | 920 | **UNKNOWN** (numeric name) |
| 3 | `6` | 9,729 | 4,737 | **UNKNOWN** (numeric name) |
| 4 | `Out of Water` | 8,160 | 7,750 | PARTIALLY VERIFIED (name only) |
| 5 | `drowning` | 4,409 | 2,650 | PARTIALLY VERIFIED (name only) |
| 6 | `swimming` | 10,737 | 4,511 | PARTIALLY VERIFIED (name only) |

- The public Roboflow page documents **three** semantic classes (`drowning`, `Out of Water`, `swimming`); the local export declares **seven**. This contradiction is unresolved.
- Numeric classes 0–3 must **not** be given semantic meaning without consistent visual evidence across multiple independent source groups.
- Do not modify the YAML or rename classes.

## D. Known dataset integrity issues

1. **15 malformed annotation lines** (all in `train`), each with 45/47/57/59/61 fields beginning with class ID 4. They are long coordinate-pair sequences (polygon-like), not valid YOLO detection lines. 15 files are affected; 6 of them contain only the malformed line and no valid box. See `DrowningDetectionTracking_v1_audit_results/invalid_annotations.csv`.
2. **1 empty label** in `test`: `test/labels/mvi_1627_vis_frame20_jpg.rf.c948cc23addbed52b173c609a43c3c59.txt` (may be a legitimate negative or a missing annotation).
3. **67 oversized boxes** covering >90% of the image: 60 class 0, 7 class 3 (likely full-frame/region boxes). See `suspicious_samples.csv`.
4. **Original `data.yaml` is missing**; `data.fixed.yaml` has an invalid absolute `path`.

## E. Leakage and duplication (VERIFIED)

- **Exact cross-split image hash groups:** 99 groups / 365 files (byte-identical images). Confirmed leakage.
- **Exact cross-split label hash groups:** 173 groups / 702 files (identical labels). Confirmed duplication.
- **Same-source-prefix groups across splits:** 685 groups / 12,667 files (frames from the same inferred source distributed across splits). Suspected temporal/source contamination, not byte-identical.
- Near-duplicate / perceptual similarity and sequential-frame identity were **not** computed.

Source grouping was derived from filenames (strip the `.rf.<hash>` suffix). See `03_source_investigation/SOURCE_GROUP_INVENTORY.csv` (2,678 groups). These are **inferred** identities, not verified video provenance.

## F. Previous visual review findings

- **Scene classification:** 3 images classified manually (index 0 `POOL`, index 1 `OUT_OF_SCOPE`, index 2 `UNCERTAIN`), all from the source clip `-Clipchamp-_mp4-10`. All other images are unreviewed. See `DrowningDetection_v1_cleaning/scene_classification_manifest.csv`.
- **Gemini visual review:** a Gemini review was referenced but **no per-tile findings were actually supplied**. Therefore **0 tile-level decisions were recorded**. Sheet 4 tiles remain `PENDING_REVIEW`.
- **Corruption check:** all 12 sheet-4 originals decode at 640×640 with normal brightness; none are black or corrupted. See `03_source_investigation/CORRUPTION_VERIFICATION_REPORT.md`.

## G. Contact sheets and exact image mappings

Directory: `E:\lifeguardDron\DrowningDetection_v1_cleaning\03_source_investigation\contact_sheets\`

| Sheet | Tiles | Focus |
|---|---|---|
| `contact_sheet_1.png` | 12 | Top-priority cross-split source groups |
| `contact_sheet_2.png` | 12 | Top-priority source groups continued |
| `contact_sheet_3.png` | 12 | Top-priority source groups continued |
| `contact_sheet_4.png` | 12 | Numeric classes 0–3 (3 tiles each) |
| `contact_sheet_5.png` | 7 | Large-box samples + the 3 already-classified images |

The authoritative tile→image mapping is:

`E:\lifeguardDron\DrowningDetection_v1_cleaning\03_source_investigation\contact_sheet_index.csv`

Columns: `sheet,split,image_name,source_group,class_ids,scene`.

### Sheet 4 exact mappings (numeric classes)

| Tile order | Split | Original image | Class |
|---|---:|---|---:|
| 1 | test | `test/images/10_MOV-52_jpg.rf.22538ea01ec12d6101cbf6612e56af2e.jpg` | 0 |
| 2 | test | `test/images/129_png_jpg.rf.97afceefe143d1f6197c661ff9d907f6.jpg` | 0 |
| 3 | test | `test/images/203512036-1-_jpg.rf.d8b66eff0d050ad589403464790ec0f1.jpg` | 0 |
| 4 | test | `test/images/0000001_02999_d_0000005_jpg.rf.64f47ee01decae0dc7efc3e9950d705e.jpg` | 1 |
| 5 | test | `test/images/0000242_00627_d_0000003_jpg.rf.f613ba88b76476f987777a023752ba6a.jpg` | 1 |
| 6 | test | `test/images/0000242_02762_d_0000010_jpg.rf.cf37ead65fef1eb83a7390acd27553a1.jpg` | 1 |
| 7 | test | `test/images/KakaoTalk_20230504_000756264_04_jpg.rf.ce823e72c6fe990c73aac4213feea6b6.jpg` | 2 |
| 8 | test | `test/images/KakaoTalk_20230504_000756264_10_jpg.rf.cc16880db7da8437d48d4ca9ef2490bc.jpg` | 2 |
| 9 | test | `test/images/KakaoTalk_20230504_000756264_16_jpg.rf.ed8afc4675aec362401c1a9985305342.jpg` | 2 |
| 10 | test | `test/images/10687_jpg.rf.edc349e55953f617f7fe8db6f6ec01f5.jpg` | 3 |
| 11 | test | `test/images/11159_jpg.rf.43800a60763e18299d65032695dc47d0.jpg` | 3 |
| 12 | test | `test/images/130163_jpg.rf.4ea8236d6b9cff5184d5670948336b77.jpg` | 3 |

## H. Images that need visual inspection (priority order)

1. **Numeric classes 0–3:** the 12 sheet-4 tiles above (highest priority — resolve class semantics).
2. **Top-priority source groups:** the 36 tiles on sheets 1–3 (judge scene and group homogeneity).
3. **Suspicious large boxes + already-classified:** the 7 tiles on sheet 5.
4. **15 malformed-label images:** inspect the original images behind `invalid_annotations.csv` (determine whether each is a real polygon/segmentation label or a labeling error).
5. **1 empty-label image:** `test/images/mvi_1627_vis_frame20_jpg.rf.c948cc23addbed52b173c609a43c3c59.jpg`.
6. **Leakage spot-checks:** for a representative subset of the 99 exact-image groups and 173 exact-label groups, open both sides and confirm they are truly identical; for a subset of the 685 source groups, confirm whether the same video/session is split across splits.

Do **not** attempt to review all 23,586 images. Review only the prioritized samples above and any images flagged by the malformed/leakage follow-ups.

## I. Visual inspection methodology

For each inspected image:

1. Open the **original** image (from the ZIP, using the path in `contact_sheet_index.csv`), not only the thumbnail.
2. Determine the **scene category**: `POOL`, `OUT_OF_SCOPE`, or `UNCERTAIN`.
3. Do not propagate a scene label to a whole source group unless every sampled tile from that group is consistent; otherwise classify per image and flag the group as mixed.
4. Record the exact image path, the scene, a short textual evidence note, and confidence.

## J. Bounding-box verification methodology

For each box drawn on an inspected image:

1. Identify the object the green box actually encloses (person, body part, whole frame, water region, object, none/background).
2. Classify box quality as one of:
   - `CORRECT` — box tightly encloses the intended object.
   - `SHIFTED` — box is offset from the object.
   - `WRONG_OBJECT` — box encloses the wrong thing.
   - `FULL_FRAME` — box covers most of the image (likely bogus).
   - `TRUNCATED` — box is cut by the image border.
   - `UNKNOWN` — cannot determine.
3. Never repair or edit a box.

## K. Class semantics investigation methodology (classes 0–3)

1. For each numeric class, inspect its sheet-4 tiles plus at least 3 independent source groups.
2. Record the consistent object identity if — and only if — the same identity appears across independent samples.
3. Confidence levels: `VERIFIED`, `PARTIALLY VERIFIED`, `UNCERTAIN`, `UNKNOWN`.
4. If evidence is inconsistent, keep the class `UNKNOWN`.

## L. Duplicate and source leakage investigation methodology

1. For exact hash groups: confirm byte-identical content on both sides.
2. For same-source-prefix groups: determine whether the same video/session frames appear in multiple splits (confirmed same-video leakage) or only share a naming pattern (suspected).
3. Classify each finding as `CONFIRMED_EXACT_DUP`, `CONFIRMED_SAME_VIDEO`, `SUSPECTED_SOURCE`, or `UNRESOLVED`.
4. Do not rebuild splits; only document evidence.

## M. Required output files and CSV schemas

All outputs go in:

`E:\lifeguardDron\DrowningDetection_v1_cleaning\03_source_investigation\`

### 1. `VISION_REVIEW_RESULTS.csv`

Columns (exact order):

`sheet,tile_index,split,image_name,source_group,class_ids,scene,box_target,box_quality,confidence,evidence,review_source,review_status`

Allowed values:

- `scene`: `POOL`, `OUT_OF_SCOPE`, `UNCERTAIN`, or empty if not scene-reviewed.
- `box_quality`: `CORRECT`, `SHIFTED`, `WRONG_OBJECT`, `FULL_FRAME`, `TRUNCATED`, `UNKNOWN`, `NO_BOX`, or empty.
- `confidence`: `VERIFIED`, `PARTIALLY VERIFIED`, `UNCERTAIN`, `UNKNOWN`, or empty.
- `review_source`: `vision-agent`.
- `review_status`: `REVIEWED`, `PENDING_REVIEW`, `UNMAPPED`.

### 2. `VISION_CLASS_SEMANTICS_REPORT.md`

For each class 0–6: class ID, YAML name, candidate meaning, evidence, number of independent source groups inspected, confidence.

### 3. `VISION_ANNOTATION_QUALITY_REPORT.md`

Sections: malformed lines (per file), oversized boxes, empty label, observed box quality by class, and whether each malformed line is repairable or must be excluded.

### 4. `VISION_DUPLICATE_AND_LEAKAGE_REPORT.md`

Sections: confirmed exact duplicates, confirmed same-video leakage, suspected source contamination, unresolved cases, with exact image pairs as evidence.

### 5. `VISION_REVIEW_SUMMARY.md`

Totals: images reviewed, scene counts, class-semantics confidence, box-quality counts, leakage classifications, remaining `PENDING_REVIEW` list, and an overall readiness statement.

### 6. `VISION_REVIEW_VISUALIZATIONS/` (only if needed)

Redrawn contact sheets or annotated evidence images. Do not overwrite originals.

## N. Execution order (manageable batches)

1. Read this document and `contact_sheet_index.csv`; confirm you can open local image files and see pixels.
2. **Batch 1:** inspect `contact_sheet_4.png` (12 tiles) → numeric class semantics + box quality.
3. **Batch 2:** inspect `contact_sheet_1.png`, `contact_sheet_2.png`, `contact_sheet_3.png` (36 tiles) → scene + group homogeneity.
4. **Batch 3:** inspect `contact_sheet_5.png` (7 tiles) → large boxes + confirm the 3 existing classifications.
5. **Batch 4:** inspect the 15 malformed-label originals.
6. **Batch 5:** inspect the 1 empty-label image.
7. **Batch 6:** spot-check exact-duplicate and same-source leakage pairs.
8. **Batch 7:** write all output files and the summary; maintain the pending-review list.

## O. Quality assurance and evidence requirements

- Every recorded decision must include the exact `image_name` and a short textual evidence note.
- Never mark an image `REVIEWED` unless its pixels were actually opened and inspected.
- Distinguish `VERIFIED`, `PREVIOUS_HUMAN`, `PREVIOUS_AI`, `INFERENCE`, `UNVERIFIED`, and `OPEN_QUESTION` in prose.
- If an image cannot be matched to `contact_sheet_index.csv`, mark it `UNMAPPED` and report it; do not guess.

## P. Safety restrictions (mandatory)

Do **not**:

- modify, delete, move, or rename original images or labels;
- change class IDs or the YAML;
- perform filtering or create a filtered dataset;
- rebuild or reassign splits;
- train any model;
- write anything into the original dataset folder or the original audit folder.

All derived outputs must stay under `DrowningDetection_v1_cleaning\03_source_investigation\`.

---

# EXECUTION PROMPT FOR THE MULTIMODAL VISION AGENT

Copy everything below this line into the vision agent.

---

You are a multimodal vision agent with genuine ability to open and inspect local image files.

Project: Autonomous AI Lifeguard System — swimming pools only.

First, read this document in full. Then confirm, in one sentence, that you can actually open local image files and inspect their pixels (not just their filenames).

Then execute these tasks in order.

## 0. Orientation

- Dataset originals (READ-ONLY): `E:\lifeguardDron\DrowningDetectionTracking_v1\` (three ZIP archives plus metadata).
- Contact-sheet mapping: `E:\lifeguardDron\DrowningDetection_v1_cleaning\03_source_investigation\contact_sheet_index.csv`.
- Contact sheets: `E:\lifeguardDron\DrowningDetection_v1_cleaning\03_source_investigation\contact_sheets\contact_sheet_1.png` through `contact_sheet_5.png`.
- Write all outputs into `E:\lifeguardDron\DrowningDetection_v1_cleaning\03_source_investigation\`.

## 1. Verify your vision capability

Open `contact_sheet_4.png` and describe the first tile in one sentence. If you cannot see pixels, stop and report that you are not a valid vision agent.

## 2. Inspect numeric classes 0–3 (contact_sheet_4.png)

For each of the 12 tiles:

1. Map the tile to its exact original image using `contact_sheet_index.csv`.
2. Open the original image (not just the thumbnail).
3. Record: scene (`POOL` / `OUT_OF_SCOPE` / `UNCERTAIN`), the object the green box encloses, box quality, and confidence.
4. Write one row per tile into `VISION_REVIEW_RESULTS.csv`.

Then, for classes 0, 1, 2, 3, write `VISION_CLASS_SEMANTICS_REPORT.md` stating the candidate meaning and confidence only if the same object identity appears across at least three independent source groups. Otherwise keep the class `UNKNOWN`.

## 3. Inspect scene and group homogeneity (sheets 1–3)

For each of the 36 tiles: record scene and confidence per tile. Do not propagate a label to a whole source group unless every sampled tile from that group is consistent; if mixed, say so in the evidence field.

## 4. Inspect sheet 5 (7 tiles)

Record scene and box quality for the large-box tiles; confirm or correct the 3 existing classifications (indices 0, 1, 2).

## 5. Malformed annotations

Open the original images behind `DrowningDetectionTracking_v1_audit_results\invalid_annotations.csv` (15 lines). For each, state whether the long coordinate sequence is a polygon/segmentation label or a corruption, and whether the image/label should be excluded. Record findings in `VISION_ANNOTATION_QUALITY_REPORT.md`. Do not edit any file.

## 6. Empty label

Inspect `test/images/mvi_1627_vis_frame20_jpg.rf.c948cc23addbed52b173c609a43c3c59.jpg`. State whether it is a legitimate negative or a missing annotation.

## 7. Duplicates and leakage

Spot-check a representative subset of the 99 exact-image groups and 173 exact-label groups (confirm identical content), and a subset of the 685 same-source groups (determine whether the same video/session appears in multiple splits). Classify each as `CONFIRMED_EXACT_DUP`, `CONFIRMED_SAME_VIDEO`, `SUSPECTED_SOURCE`, or `UNRESOLVED`. Write `VISION_DUPLICATE_AND_LEAKAGE_REPORT.md`.

## 8. Final outputs

Write:

1. `VISION_REVIEW_RESULTS.csv`
2. `VISION_CLASS_SEMANTICS_REPORT.md`
3. `VISION_ANNOTATION_QUALITY_REPORT.md`
4. `VISION_DUPLICATE_AND_LEAKAGE_REPORT.md`
5. `VISION_REVIEW_SUMMARY.md`
6. `VISION_REVIEW_VISUALIZATIONS/` only if needed

Follow the exact CSV schema in section M of this document. In `VISION_REVIEW_SUMMARY.md`, report totals, remaining `PENDING_REVIEW` items, and an overall readiness statement. Never claim an image was reviewed unless you actually opened it.

## Safety

Do not modify, delete, or move original images or labels. Do not change class IDs or the YAML. Do not filter, split-rebuild, or train. Write only into the derived investigation directory.
