# NEXT PHASE EXECUTION PLAN

Date: 2026-11-04  
Target directory: `05_recovery_execution/10_verified_pool_cohort/` (new; not created yet)

## Objective

Create a **small, trustworthy, pool-only person-detection dataset** with completed human-verified annotations and a valid source-level split. Do not train in this phase.

## Phase 4A — Freeze and correct split metadata

Tasks:
1. Freeze `VALIDATED_SPLIT_PROPOSAL.csv` as invalid for training.
2. Produce `SPLIT_STATUS_V2.csv` with one row per image and only these usable statuses:
   - `READY_VERIFIED_GOLD` (2)
   - `READY_VERIFIED_CLEAN` (40)
   - everything else `NOT_READY_*` with reason.
3. Remove all HOLD/WITHHOLD/REJECTED rows from active splits.
4. Recompute the frame cap on the READY-only cohort.

Acceptance:
- Active rows = READY only (42 initially, then the new cohort).
- 0 rows with `phase3_split_eligibility` other than READY in active splits.
- Frame cap: 0 groups > 60 active frames.

## Phase 4B — Source-unit shortlist

Candidate clean source families (from existing reports; not yet approved):
- `SF02_output_NNNN` `output_` single-person frames (693 P1 rows; 680 in one source group — must cap).
- `SF06_kakaotalk` (176 P1 rows).
- `SF10_video_2022_12_07` (248 P1 rows; 100-frame group must cap).
- `SF11_swim_ad_mp4` (129 unique SHAs; competitive indoor pool).
- `SF12_clipchamp_mp4` (47 P1 rows; outdoor pool).
- `SF02_output_NNNN` `output1..90` (the 42 READY seed).

Exclude from the first batch:
- `SF03_imageNNNN` (Thai TV broadcast with permanent chyron/PiP).
- `SF18_other_static` (watermarked stock photos).
- `SF08_drowningN_mp4` mixed open-water clips until scene-separated.
- `SF04_pure_digits` synthetic sprite composites.
- Any frame with black corners, cutout boxes, or box-level shear seams.

Acceptance:
- At least 4 independent source units selected.
- No source unit contributes >60 frames to the cohort.
- Every selected source is confirmed pool by human visual review.

## Phase 4C — Reannotation (single class `person`)

Tasks:
1. For each selected image, draw boxes around **every visible person** (in water and on deck/rescue).
2. Use a single class ID `0: person` for the detector.
3. Do not label drowning/swimming/out-of-water in this dataset; behavior belongs to the temporal stage.
4. Reject any image with unresolved occlusion, merged people, or corrupt pixels.
5. Record every decision in `VERIFIED_POOL_COHORT.csv`.

CSV schema:
`image_id,split,source_unit,rel_image_path,image_sha256,width,height,boxes_yolo,box_count,annotator,review_status,review_notes`

Acceptance:
- 100% of cohort images visually reviewed.
- 0 merged multi-person boxes.
- 0 missing visible people in a 20% random re-check.
- 0 out-of-scope/corrupt/synthetic images.

## Phase 4D — Split V2 and QA

Tasks:
1. Split by source unit, not by image.
2. Ensure 0 exact SHA duplicates and 0 source-unit leakage.
3. Apply ≤60 frames per source.
4. Create `data_v2.yaml` with `nc: 1`, `names: [person]`.
5. Produce overlay sheets per source unit and QA tables.

Acceptance:
- `SPLIT_V2.csv` with active splits only from `VERIFIED_POOL_COHORT.csv`.
- Leakage checks pass on V2, not on the old split.
- QA: 0 invalid YOLO lines; 0 boxes outside [0,1]; 0 empty labels unless intentional negative (documented).
- Frame cap 0 violations.

## Phase 4E — Freeze

- Freeze the cohort and split.
- Do not train.
- Report counts, source units, class schema, and remaining limitations.

## Tools

- Python 3.11, stdlib, Pillow/OpenCV for overlays.
- Manual annotation tool (e.g., CVAT) for box drawing; no installation without approval.
- No model training, no dataset download.

## Evidence of success

- `VERIFIED_POOL_COHORT.csv` (all `REVIEWED`).
- `SPLIT_V2.csv` + `SPLIT_V2_REPORT.md` (0 leakage, 0 cap violations).
- Overlay contact sheets per source unit.
- `QA_V2.csv` and `data_v2.yaml`.
- A clear statement of cohort size, source-unit diversity, and unresolved limitations.

## Risks

- Reannotation effort may be larger than expected; start with 300–500 images, not all 2,924.
- Source families may be less diverse than their names suggest; verify video identity, not filenames.
- Single-class `person` loses behavior semantics; that is intentional for the detector.
- Do not let the old split or the 3,522 manifest leak back into training.
