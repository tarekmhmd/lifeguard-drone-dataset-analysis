# Class Investigation Report — Numeric Classes 0–3 (DrowningDetectionTracking_v1)

Date: 2026-11-02
Status: visual investigation **in progress**. No class meaning is asserted without evidence.

## Method

- Class instances and image counts are taken from `class_distribution.csv` (full audit).
- Representative images for each numeric class were sampled across distinct source groups and rendered with bounding boxes in the contact sheets.
- A Gemini visual-audit report was referenced as preliminary human-review evidence, but **no per-tile findings were supplied in this session**, so no tile-level decisions were recorded.

## Findings

| Class ID | YAML name | Boxes | Images | Meaning | Confidence |
|---:|---|---:|---:|---|---|
| 0 | `3` | 9,890 | 8,597 | Unknown | UNKNOWN |
| 1 | `4` | 4,683 | 370 | Unknown | UNKNOWN |
| 2 | `5` | 1,432 | 920 | Unknown | UNKNOWN |
| 3 | `6` | 9,729 | 4,737 | Unknown | UNKNOWN |

## Observations (not yet verified)

- Numeric classes 0 and 3 are frequent and co-occur with semantic classes in large `youtube-*` source groups, suggesting they are real annotation categories, not unused YAML slots.
- Class 1 is sparse in images (370 images) despite 4,683 boxes.
- Large (>90% area) boxes are concentrated in classes 0 (60) and 3 (7).
- The numeric names give no semantic information; the README contains no class definitions.

## Review status

- 12 representative numeric-class tiles exist in `contact_sheet_4.png`.
- **0 tile-level decisions recorded** (findings not supplied).
- All 12 tiles are marked **PENDING_REVIEW** in `SCENE_CLASSIFICATION_PROGRESS.csv`.
- Classes 0–3 remain **UNKNOWN**; no mapping is asserted.
