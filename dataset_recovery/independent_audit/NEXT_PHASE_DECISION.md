# NEXT PHASE DECISION

Date: 2026-11-04  
Based on: `INDEPENDENT_PHASE3_AUDIT.md`, `COUNT_RECONCILIATION.csv`, `DATASET_READINESS_ASSESSMENT.csv`, `UNRESOLVED_ISSUES.csv`

## Decision

**B — Continue targeted recovery, but narrowed to a small, verified, source-grouped pool cohort. Do not proceed with the current split and do not train.**

Specifically: **Phase 4 will build a new small detection cohort (single `person` class) with completed human-verified reannotation and a source-level split, instead of trying to rescue all 3,522 reannotation rows or using `VALIDATED_SPLIT_PROPOSAL.csv` as-is.**

## 1. Most important blocker

**There is no trustworthy annotated pool cohort.**

- Only **42** images are READY (2 gold + 39 accept + 1 correction).
- **3,038 of 3,522** reannotation rows have no proposed label at all.
- The **484** proposed labels are class-ID remaps only; coordinates are unchanged and every one is `REQUIRES_VISUAL_BOX_POSITION_CHECK_BEFORE_USE`.
- **2,924** pending images are not reannotated and are still assigned to active splits.
- Class semantics are source-specific and inconsistent (`drowning`/`swimming`/`Out of Water` shift by family).

The **split defects are a symptom** of this: `VALIDATED_SPLIT_PROPOSAL.csv` has 4,218 active rows, but only 42 are READY; 4,176 are pending/withheld/rejected. The 7-frame-cap failure is also downstream.

## 2. Why this must come before anything else

Training on the current artifacts would:

- supervise boxes for people that were never reannotated,
- mix behavior labels whose meaning changes by source,
- include corrupted/out-of-scope/pending images,
- produce metrics that look like progress but are not trustworthy.

For a safety-critical lifeguard aid, a small trustworthy set is more valuable than a large unreliable one.

## 3. Exact affected files and cohorts

- `05_recovery_execution/06_split_validation/VALIDATED_SPLIT_PROPOSAL.csv` — 4,176/4,218 active rows not READY; 1,895/1,935 capped rows not READY.
- `05_recovery_execution/04_reannotation_3463/REANNOTATION_MANIFEST.csv` — 3,038 rows with no proposed label.
- `05_recovery_execution/04_reannotation_3463/ANNOTATION_CORRECTIONS.csv` — 484 class-only remaps; unverified boxes.
- `05_recovery_execution/05_human_review_1813/HUMAN_REVIEW_QUEUE.csv` — 1,827 unresolved.
- `05_recovery_execution/06_split_validation/FRAME_CAP_AUDIT.csv` — 7 cap violations; `SPLIT_QA.csv` LK_04 FAIL vs `LEAKAGE_RECHECK.csv` PASS.
- `02_gold_reference_set/GOLD_REFERENCE_REVIEW.csv` + `03_label_review_737/LABEL_REVIEW_737.csv` — the 42 READY images, from only 2 video units.

## 4. What the next phase must accomplish

1. Freeze the current split as invalid; do not consume its `proposed_split` values.
2. Build a **new verified pool cohort** from a limited set of clean, unrotated, real pool video sources.
3. Reannotate **all visible people** into a **single detection class `person`** (not `drowning`/`swimming`). Keep behavior labels out of the detector; they belong to the temporal stage.
4. Build a **source-level split** (train/valid/test by source unit), capped at ≤60 frames per source, with zero exact/source leakage.
5. Produce a conservative initial dataset only after human visual verification.

## 5. Tools and scripts

- Python 3.11 + stdlib + Pillow/OpenCV for overlays and QA; no training.
- Existing CSV/hash tooling under `05_recovery_execution/scripts/` as reference only.
- A manual annotation tool (e.g., CVAT) may be used for drawing boxes; do not install without approval.
- All outputs under `05_recovery_execution/10_verified_pool_cohort/` (new).

## 6. Acceptance criteria

Proposed, to be confirmed with the supervisor where noted:

- Cohort size: target **300–500** images, from **≥4 independent video units** (PROPOSED; not a fact).
- 100% of cohort images opened and visually verified; 0 `SOURCE_INFERRED` images in the final training set.
- Every visible person in the pool and on the deck annotated; no merged multi-person boxes; no missing border swimmers.
- Single class `person` only; no numeric/behavior classes in the detector labels.
- No synthetic sprites, cutout boxes, rotation wedges, sheared box patches, or out-of-scope scenes.
- Source-level split: 0 exact SHA duplicates and 0 source-unit leakage across splits; ≤60 frames per source.
- A new local `data.yaml`; original `data.yaml`/`data.fixed.yaml` untouched.
- Human review of all validation/test images before training.

## 7. Evidence that the phase succeeded

- `VERIFIED_POOL_COHORT.csv` with path, split, source unit, SHA, boxes, review status = `REVIEWED`.
- `SPLIT_V2.csv` / `SPLIT_V2_REPORT.md` showing 0 SHA/source leakage and cap compliance.
- Overlay sheets per source unit with boxes drawn and human sign-off.
- `QA_V2.csv` with 0 invalid boxes, 0 unannotated visible people (sampled audit), 0 out-of-scope.
- A clean `data_v2.yaml`.
- No training until the above pass.

## Why not A or C

- **Not A (proceed with filtering):** the current split is not a valid filter input; 4,176/4,218 active rows are not READY, and 7 cap violations exist.
- **Not C (stop using the dataset):** the dataset is not fundamentally unusable; 42 verified images and several clean source families exist. But it cannot be used at the scale or with the labels currently claimed. The correct move is a small, verified recovery, not silent rejection.
