# Forensic Dataset Audit — DrowningDetectionTracking_v1

Audit date: 2026-11-02
Scope: the local audit package under `E:\lifeguardDron\DrowningDetectionTracking_v1` only.
Original archives were treated as read-only. No dataset file was modified, renamed, deleted, or overwritten.

## 1. Evidence status

**VERIFIED FACTS**

- Three ZIP archives open successfully and pass `ZipFile.testzip()`.
- 23,586 images and 23,586 labels are present.
- Every image decodes; all are 640×640; no zero-byte or unsupported image files.
- Image/label pairing is complete: 0 missing labels, 0 orphan labels, 0 duplicate stems.
- 49,040 valid YOLO detection boxes were parsed.
- 15 malformed annotation lines exist in train; 0 in valid/test.
- One empty label file exists in test.
- 67 boxes cover more than 90% of the image area; 60 are class 0 and 7 are class 3.
- Exact cross-split image duplicates exist: 99 hash groups containing 365 files.
- Exact cross-split label duplicates exist: 173 hash groups containing 702 files.
- All seven class IDs (0–6) occur in every split.
- The labels contain five fields only; no track IDs, frame numbers, or video IDs.

**WARNINGS**

- The dataset mixes two annotation schemas: numeric classes 0–3 and semantic classes 4–6.
- Numeric classes are not documented.
- 6 train label files contain only malformed lines and no valid detection.
- The original `data.yaml` is missing from the package; only `data.fixed.yaml` is present.
- `data.fixed.yaml` contains a Colab-specific absolute `path` that is not valid for this local folder.
- Confirmed exact duplicate images and labels cross train/valid/test.
- 685 source-prefix groups occur across splits, indicating suspected temporal/source contamination.

**UNVERIFIED ASSUMPTIONS**

- The semantic meaning of classes 0–3 and of `Out of Water`, `drowning`, and `swimming`.
- Whether numeric classes are placeholders or meaningful categories.
- Visual semantic correctness of the boxes.
- Whether the empty test label represents a legitimate negative image.
- Near-duplicate/perceptual image similarity and sequential-video-frame identity.
- The original contents of the missing `data.yaml`.

## 2. Package contents

Expected files were not all present under the expected names.

| Expected | Actual |
|---|---|
| `train_labels.zip` | `train-20261102T163226Z-1-001.zip` |
| `valid_labels.zip` | `valid-20261102T163228Z-1-001.zip` |
| `test_labels.zip` | `test-20261102T163222Z-1-001.zip` |
| `README.roboflow.txt` | present |
| `README.dataset.txt` | present |
| `data.yaml` | **missing** |
| `data.fixed.yaml` | present |

The actual ZIPs contain both images and labels, not label-only archives.

## 3. Metadata forensics

- Source: Roboflow project `drowningdetection-2zzxz`, project `drowningdetectiontracking`, version 1.
- Export page: `https://universe.roboflow.com/drowningdetection-2zzxz/drowningdetectiontracking/dataset/1`.
- License: CC BY 4.0.
- README reports 23,586 images, YOLOv11 format, resize to 640×640 (stretch), and three augmented versions per source image.
- README provides **no class definitions** and **no tracking definitions**.

`data.fixed.yaml`:

- `nc: 7`
- `names: ['3', '4', '5', '6', 'Out of Water', 'drowning', 'swimming']`
- `train: train/images`
- `val: valid/images`
- `test: test/images`
- `path:` points to `/content/drive/.shortcut-targets-by-id/.../DrowningDetectionTracking_v1`

The relative split paths are valid for the package root. The absolute `path` is not valid on this Windows audit copy. The original `data.yaml` is absent, so its contents cannot be compared with the fixed YAML.

`nc` matches the number of declared names. Class IDs 0–6 are consistent with the seven-name list, but the semantics of IDs 0–3 are unknown.

## 4. ZIP archive integrity

| Split | Archive | Size (bytes) | `testzip` | Entries | JPG | TXT | Duplicate entries | Unsupported files |
|---|---|---:|---|---:|---:|---:|---:|---:|
| train | `train-20261102T163226Z-1-001.zip` | 977,915,359 | None | 42,738 | 21,369 | 21,369 | 0 | 0 |
| valid | `valid-20261102T163228Z-1-001.zip` | 48,579,536 | None | 2,430 | 1,215 | 1,215 | 0 | 0 |
| test | `test-20261102T163222Z-1-001.zip` | 43,501,365 | None | 2,004 | 1,002 | 1,002 | 0 | 0 |

Internal structure is consistent: each archive contains `<split>/images` and `<split>/labels`.

## 5. Split-level annotation integrity

| Split | Images | Labels | Valid boxes | Empty labels | Invalid lines | Invalid files | Only-invalid files | Out-of-bounds | Duplicate lines | Tiny boxes | Large boxes | Missing labels | Orphan labels |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 21,369 | 21,369 | 44,834 | 0 | 15 | 15 | 6 | 0 | 0 | 0 | 60 | 0 | 0 |
| valid | 1,215 | 1,215 | 2,655 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 0 | 0 |
| test | 1,002 | 1,002 | 1,551 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 |
| **Total** | **23,586** | **23,586** | **49,040** | **1** | **15** | **15** | **6** | **0** | **0** | **0** | **67** | **0** | **0** |

- Invalid annotation rate: 15 / 49,055 total non-empty lines = **0.0306%**.
- Empty label rate: 1 / 23,586 = **0.0042%** (test: 1 / 1,002 = 0.0998%).
- Large-box rate: 67 / 49,040 = **0.1366%**.
- No invalid class IDs, no invalid coordinates, no duplicate annotation lines, no tiny boxes, no out-of-bounds boxes.

### Malformed lines

All 15 malformed lines are in train. They contain 45, 47, 57, 59, or 61 fields:

- 45 fields: 3 lines
- 47 fields: 3 lines
- 57 fields: 3 lines
- 59 fields: 3 lines
- 61 fields: 3 lines

All begin with class ID 4 and consist of long normalized-coordinate sequences. They are consistent with polygon/segmentation data in a detection-label file, not with valid YOLO detection lines.

9 of the 15 affected files also contain valid detection lines. 6 files contain only malformed lines and therefore have no valid annotations.

## 6. Class distribution

| ID | Name | Train boxes | Train images | Valid boxes | Valid images | Test boxes | Test images | Total boxes | % of boxes |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | `3` | 9,032 | 7,856 | 546 | 466 | 312 | 275 | 9,890 | 20.17% |
| 1 | `4` | 4,111 | 342 | 442 | 19 | 130 | 9 | 4,683 | 9.55% |
| 2 | `5` | 1,322 | 850 | 84 | 54 | 26 | 16 | 1,432 | 2.92% |
| 3 | `6` | 8,773 | 4,218 | 583 | 276 | 373 | 243 | 9,729 | 19.84% |
| 4 | `Out of Water` | 7,375 | 6,978 | 387 | 384 | 398 | 388 | 8,160 | 16.64% |
| 5 | `drowning` | 4,123 | 2,434 | 172 | 113 | 114 | 103 | 4,409 | 8.99% |
| 6 | `swimming` | 10,098 | 4,234 | 441 | 186 | 198 | 91 | 10,737 | 21.89% |
| **Total** | | **44,834** | | **2,655** | | **1,551** | | **49,040** | **100%** |

### Schema mix

| Split | Total images | Numeric-only | Semantic-only | Mixed numeric+semantic | No valid class | Empty |
|---|---:|---:|---:|---:|---:|---:|
| train | 21,369 | 9,792 | 10,968 | 603 | 6 | 0 |
| valid | 1,215 | 570 | 600 | 45 | 0 | 0 |
| test | 1,002 | 436 | 540 | 25 | 0 | 1 |

Numeric classes 0–3 account for **23,238 boxes (47.4%)**. Semantic classes 4–6 account for **25,802 boxes (52.6%)**.

Multi-class images: 4,760 train, 255 valid, 108 test.

### Imbalance

- Class 1 (`4`) appears in only **1.60%** of train images, **1.56%** of valid, and **0.90%** of test.
- Class 2 (`5`) appears in **3.98%** train, **4.44%** valid, **1.60%** test.
- Class 0 (`3`) is the most widespread image class: 36.76% train, 38.35% valid, 27.45% test.
- Class 6 (`swimming`) drops from 19.81% train images to 15.31% valid and 9.08% test.
- All seven classes occur in all three splits, but the image-level imbalance for classes 1 and 2 is severe.

## 7. Image integrity and pairing

| Split | Images | Decoded | Corrupt | Zero-byte | Unsupported | 640×640 | Other sizes | Missing labels | Orphan labels | Duplicate image stems |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 21,369 | 21,369 | 0 | 0 | 0 | 21,369 | 0 | 0 | 0 | 0 |
| valid | 1,215 | 1,215 | 0 | 0 | 0 | 1,215 | 0 | 0 | 0 | 0 |
| test | 1,002 | 1,002 | 0 | 0 | 0 | 1,002 | 0 | 0 | 0 | 0 |

All images are square 640×640 (aspect ratio 1.0). No image corruption or unsupported format was detected. Image/label pairing is complete.

## 8. Duplicate and leakage findings

| Finding | Groups | Files | Status |
|---|---:|---:|---|
| Exact SHA-256 image duplicates across splits | 99 | 365 | **Confirmed leakage** |
| Exact SHA-256 label duplicates across splits | 173 | 702 | **Confirmed duplication** |
| Same image filename across splits | 0 | 0 | None |
| Same label filename across splits | 0 | 0 | None |
| Same source-prefix groups across splits | 685 | 12,667 | **Suspected temporal/source contamination** |

Examples of exact cross-split image duplicates include `youtube-6`, `youtube-5`, `youtube-3`, `youtube-4`, `youtube-2`, `video40-1-_mp4-144`, `video40-2-_mp4-36`, and `video_2022-12-07_23-08-12_mp4-14`. These images appear in both train and valid/test, sometimes with three train augmentations.

The 685 source-prefix groups are not proof of identical frames, but they show that frames from the same source video are distributed across splits. This is a contamination risk for temporal/video-derived data.

Near-duplicate/perceptual hashing and sequential-frame identity were not performed.

## 9. Visual verification

Contact sheets were generated:

- `visual_samples/classes_0_3.png`
- `visual_samples/classes_4_6_and_suspicious.png`

The audit environment could not display the images to the auditor, so **visual semantic correctness remains NOT VERIFIED**. No claim is made that boxes surround the intended objects.

Geometry-based warning: 67 boxes cover more than 90% of the image area. 60 are class 0 (`3`) and 7 are class 3 (`6`). 54 of the 60 class-0 large boxes are in train; 5 more are in valid; 1 is in test. These are likely full-frame or near-full-frame boxes and require visual review.

The single empty test label belongs to `mvi_1627_vis_frame20_jpg.rf.c948cc23addbed52b173c609a43c3c59`; it may be a legitimate negative image or a missing annotation. Visual confirmation is required.

## 10. Tracking and semantic limitations

- Label format: five columns only (`class_id x_center y_center width height`).
- No object IDs, track IDs, frame numbers, or video IDs are present.
- The dataset name contains “Tracking,” but the annotations are object detection annotations, not tracking annotations.
- README provides no definitions for `Out of Water`, `drowning`, `swimming`, or numeric classes `3`–`6`.
- The numeric and semantic schemas are mixed; numeric-only images are a large fraction of the dataset.
- Without documentation, the meaning of classes 0–3 is unknown. They cannot be assumed to be placeholders or semantic classes.

## 11. What could not be verified

- Visual confirmation that boxes align with objects.
- Visual confirmation that `drowning` frames show genuine drowning behavior.
- Semantic meaning of numeric classes.
- Whether the empty test label is a legitimate negative.
- Near-duplicate/perceptual similarity and sequential-frame leakage.
- Original `data.yaml` contents.
- Whether the combined numeric/semantic class schema was intentional.
- Whether `Out of Water`, `drowning`, and `swimming` are mutually exclusive or overlapping.

## 12. Final decision

**Overall verdict: CLEAN AND RE-EVALUATE.**

The dataset contains a large, structurally valid detection set, but it is not safe to use as-is for the Lifeguard Drone’s safety-critical drowning task. The problems are identifiable and potentially fixable; the dataset is not fundamentally unusable.

| Intended task | Assessment |
|---|---|
| Swimmer detection | **KEEP WITH LIMITATIONS after cleaning.** Class 6 `swimming` provides 10,737 boxes; class 4 `Out of Water` provides 8,160. Numeric schema, leakage, and large boxes must be cleaned first. |
| Drowning detection | **SUPPLEMENT / NOT APPROVED as-is.** Class 5 has 4,409 boxes, but the semantic validity of `drowning` is unverified and the dataset has confirmed leakage and mixed class schema. |
| Temporal distress classification | **SUPPLEMENT.** No temporal labels, frame IDs, track IDs, or distress states. Source-prefix grouping exists, but leakage undermines video-level evaluation. |
| Multi-object tracking | **SUPPLEMENT.** No track IDs or object identities. Useful only for detection pretraining, not for training a tracker. |

### Why not KEEP

- Confirmed cross-split exact image and label duplicates.
- 15 malformed polygon-like lines; 6 files have no valid annotation.
- 67 large/full-frame boxes.
- Mixed numeric/semantic schema with undocumented classes.
- Severe image-level imbalance for classes 1 and 2.
- Missing original `data.yaml` and ambiguous class semantics.

### Why not REJECT

- 23,586 images and 23,586 labels are present.
- ZIPs are intact; all images decode; all are 640×640.
- Image-label pairing is complete.
- 49,040 boxes parse as valid five-field YOLO detections.
- All seven classes occur in all splits.
- The data can support detection pretraining after cleaning.

## 13. Recommended corrective actions

1. **Fix or quarantine the 15 malformed lines**; the 6 files containing only malformed lines must be repaired or excluded.
2. **Resolve the class schema.** Document or remove numeric classes 0–3. Do not train a safety-critical model with undocumented classes.
3. **Remove or repair the 67 large/full-frame boxes**, especially class 0 and class 3, after visual review.
4. **Repair split leakage** using exact hashes and source-prefix grouping; move all duplicate/source-related frames to one split.
5. **Restore `data.yaml`** or create a verified local YAML; remove the invalid absolute `path` from the fixed YAML.
6. **Visually review** representative samples for every class and the empty-label image; do not approve `drowning` semantics without evidence.
7. **Supplement** drowning and temporal/tracking data if those tasks remain in scope.
8. **Do not train** a safety-critical drowning model on this dataset as-is.

## 14. Deliverables

- `audit_report.md`
- `class_distribution.csv`
- `split_integrity.csv`
- `invalid_annotations.csv`
- `suspicious_samples.csv`
- `audit_summary.txt`
- `image_integrity.csv`
- `duplicate_images.csv`
- `same_source_prefix.csv`
- `label_summary.json`
- `image_summary.json`
- `visual_samples/classes_0_3.png`
- `visual_samples/classes_4_6_and_suspicious.png`
