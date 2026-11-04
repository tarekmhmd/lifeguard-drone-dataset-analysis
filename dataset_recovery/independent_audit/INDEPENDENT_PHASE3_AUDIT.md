# Independent Phase 3 Audit — DrowningDetectionTracking_v1

Date: 2026-11-04  
Auditor role: independent dataset recovery auditor  
Scope: read-only inspection of Phase 3 artifacts. No original data modified.

## 1. Evidence actually inspected

Reports:
- `00_state/EXECUTION_STATE.md`, `00_state/EXECUTION_LOG.md`
- `01_evidence_validation/EVIDENCE_VALIDATION_REPORT.md`
- `02_gold_reference_set/GOLD_REFERENCE_REPORT.md`
- `03_label_review_737/LABEL_REVIEW_REPORT.md`, `AUTONOMOUS_PHASE3C_FINAL_REPORT.md`
- `04_reannotation_3463/REANNOTATION_REPORT.md`
- `05_human_review_1813/HUMAN_REVIEW_REPORT.md`
- `06_split_validation/SPLIT_VALIDATION_REPORT.md`
- `07_quality_assurance/FINAL_QA_REPORT.md`
- `08_reports/PHASE3_DATASET_RECOVERY_FINAL_REPORT.md`
- Phase 2: `RECOVERY_DECISION_REPORT.md`, `DATASET_RECOVERY_FINAL_REPORT.md`, `ANNOTATION_FORENSICS_REPORT.md`, `DUPLICATE_AND_LEAKAGE_FULL_REPORT.md` (as referenced)

CSV/data:
- `LABEL_REVIEW_737.csv`, `REANNOTATION_MANIFEST.csv`, `ANNOTATION_CORRECTIONS.csv`, `HUMAN_REVIEW_QUEUE.csv`, `GOLD_REFERENCE_REVIEW.csv`
- `VALIDATED_SPLIT_PROPOSAL.csv`, `SPLIT_VALIDATION.csv`, `LEAKAGE_RECHECK.csv`, `FRAME_CAP_AUDIT.csv`
- `IMAGE_QA.csv`, `ANNOTATION_QA.csv`, `SEMANTIC_QA.csv`, `SPLIT_QA.csv`
- `SOURCE_CLASS_MAPPING.csv`
- `proposed_labels/` (968 files = 484 orig/proposed pairs)

Scripts executed by this audit (read-only):
- CSV reconciliation and membership-set comparison.
- SHA-256 verification of 737 image pairs, 737 label pairs, 484 correction images, and 42 clean images directly from the original ZIPs.
- Correction-pair diffing (class vs coordinate vs box-count changes).
- Active-split eligibility/leakage/frame-cap recomputation.

## 2. 8.1 What is genuinely complete

- **737 label-review cohort:** exactly 737 unique paths, exactly matching the Phase 2 `KEEP_IMAGE_REVIEW_LABEL` representative set; 0 missing, 0 extra. All 737 image SHA and 737 label SHA verified against the original archives.
- **3,522 reannotation manifest:** exactly 3,463 Phase 2 + 59 Phase 3C REANNOTATE; 3,522 unique paths and 3,522 unique SHAs; 0 missing/extra.
- **1,827 human-review queue:** exactly 1,813 Phase 2 + 14 Phase 3C HUMAN_REVIEW; 1,827 unique paths/SHAs; 0 overlap with the 42 clean images.
- **42 clean images:** 2 `VERIFIED_GOLD` + 39 `ACCEPT_AS_IS` + 1 `ACCEPT_AFTER_VERIFIED_CORRECTION`; 42 unique paths and SHAs verified.
- **484 correction pairs:** 484 unique paths, 484 orig/proposed pairs; all SHA-verified.
- **Split leakage on active rows:** 0 exact `sha_group_id` duplicates and 0 `source_group` leakage across the active `PROPOSED_*` rows.
- **Original data untouched:** no original ZIP/image/label modification was found.

## 3. 8.2 What is partially complete

- **Reannotation:** 484 of 3,522 rows have proposed label files. However, all 484 are **class-ID remaps only**:
  - 216 `0→5`, 234 `3→5`, 34 `2→5`.
  - Coordinates are identical in all 484; no new boxes; no box-count changes.
  - All 484 retain `REQUIRES_VISUAL_BOX_POSITION_CHECK_BEFORE_USE`.
  - No proposed file was applied to the original or promoted to final.
- **Visual review:** 355/737 (48.17%) are `VERIFIED_VISUAL`; 382/737 (51.83%) are `SOURCE_INFERRED`. The reported "100%" is **unique base-prefix coverage**, not per-image visual coverage.
- **Human review:** the queue is complete as a queue, but 1,827 decisions are not resolved.
- **Split validation:** exact/source leakage checks pass on the active rows, but the active rows themselves are invalid, and the frame-cap check fails.

## 4. 8.3 What is not complete

- **3,038 reannotation rows have no proposed label file.**
- **2,924 P1 pending images** are not reannotated and are still assigned active `PROPOSED_*` splits.
- **14 Phase3C HUMAN_REVIEW rows** are still active in `PROPOSED_TRAIN`.
- **1,827 human-review cases** are unresolved.
- **Class semantics** are not unified; `SOURCE_CLASS_MAPPING.csv` shows source-specific shifts, and class 5/6 meanings conflict.
- **No valid training split** exists: only 42 of 4,218 active rows are READY.
- **No training-ready dataset** exists.

## 5. 8.4 Count reconciliation

Full table: `COUNT_RECONCILIATION.csv`. Key results:

| Item | Earlier | Latest reported | Independently verified | Difference explained? |
|---|---:|---:|---:|---|
| 737 review candidates | 737 | 737 | **737** | YES |
| 3,463 original reannotation | 3,463 | 3,522 | **3,522** | YES (3,463+59) |
| 1,813 original human review | 1,813 | 1,827 | **1,827** | YES (1,813+14) |
| 42 clean images | 42 | 42 | **42** | YES, but only 2 video units |
| 2,924 pending | 2,924 | 2,924 | **2,924** | Membership YES; "1,615 unassigned" **NO** |
| 484 corrections | 484 | 484 | **484** | YES, but class-remap only |
| Active split rows | — | 4,218 | **4,218** | 42 READY, 4,176 invalid |
| Capped subset | — | 1,935 | **1,935** | 40 READY, 1,895 invalid |
| Frame-cap violations | — | 0 claimed | **7** | NO |

Batch counts: authoritative `LABEL_REVIEW_737.csv` has **51 / 42 / 43 / 601**; the earlier narrative **39 / 40 / 86 / 572** was a pre-filter draft and does not describe the final membership. Membership is exact.

## 6. 8.5 Dataset readiness

| Area | Verdict | Evidence |
|---|---|---|
| Image integrity | **READY** | 23,586 decode; all 640×640; no corrupt/zero/unsupported; 737+484+42 SHA verified |
| Pool-only scope | **NOT READY** | Only 42 READY; 2,924 pending include broadcast/stock/mixed; 85 out-of-scope + 539 corrupted active |
| Annotation correctness | **BLOCKED** | 484 class-only remaps unverified; 3,038 no proposed labels; class shifts |
| Annotation completeness | **NOT READY** | 42 complete; 3,038 pending/no proposed; 2,924 P1 pending |
| Class consistency | **BLOCKED** | Source-specific shifts; 5/6 inconsistent; no global mapping |
| Duplicate control | **CONDITIONAL** | Active rows 0 exact/source leakage; prior audit 99/173/685 groups |
| Video/source leakage | **CONDITIONAL** | Active source leakage 0; 7 frame-cap violations; 685 same-source groups |
| Split suitability | **NOT READY** | 4,176/4,218 active invalid; 1,895/1,935 capped invalid; 7 cap fails |
| **Overall training readiness** | **BLOCKED** | No trustworthy labels; no valid split; only 42 clean images |

## 7. Critical discrepancies discovered

1. **Active split contains non-ready data.** `VALIDATED_SPLIT_PROPOSAL.csv` has 4,218 active rows, but only 42 are `READY_VERIFIED_GOLD/CLEAN`. 2,924 are `HOLD_PENDING_REANNOTATION`; 592 are `WITHHOLD_ROTATED_AUGMENTED_VARIANT`; 660 are rejected/withheld. The split report’s per-split table (1,611/290/267) does not match the CSV’s proposed_split counts (2,890/664/664).
2. **Capped subset is mostly invalid.** 1,935 rows in `in_balanced_capped_subset`; only 40 READY; 1,895 are HOLD/WITHHOLD/REJECTED.
3. **Frame-cap contradiction.** `FRAME_CAP_AUDIT.csv` has 7 non-compliant groups; `SPLIT_QA.csv` records `LK_04 FAIL`; `LEAKAGE_RECHECK.csv` records `PASS`. The final report and execution state claim 0 violations.
4. **2,924 pending rows are not unassigned.** All have `PROPOSED_TRAIN/VALID/TEST`; the split report claims 1,615 unassigned.
5. **Reannotation is not complete.** 3,038 rows have no proposed label; 484 proposed files change only class IDs, not boxes.
6. **42 clean images are not independent.** 40 come from one source group (`output`, SF02 indoor pool); 2 come from `130154/130155` (SF04 overhead). Only two video units.
7. **Full visual coverage is overstated.** 355/737 individually `VERIFIED_VISUAL`; 382 `SOURCE_INFERRED`; 100% is base-prefix coverage.

## 8. Annotation-quality findings

- 15 malformed polygon-like lines (15 train files; 6 only-invalid) are **excluded from the active split** (routed to HUMAN_REVIEW or DUPLICATE/OUT_OF_SCOPE). They remain in the original archive.
- 1 empty test label is `MVI_1627_VIS_frame20`, an **OUT_OF_SCOPE maritime** image; excluded from active split.
- 67 oversized boxes: 60 class 0, 7 class 3. 9 are WITHHOLD, 5 HOLD; none are READY.
- No invalid class IDs, invalid coordinates, out-of-bounds boxes, or duplicate annotation lines were found in the full audit.
- Class semantics remain inconsistent by source family; no deterministic global mapping is safe.

## 9. Visual-inspection limitations

- This independent auditor did **not** open image pixels. Visual inspection claims are therefore **not independently verified** by this audit.
- The prior reports claim direct visual review of contact sheets and overlays (18 gold, 24 contact sheets / 387 tiles, 25 overlays, 64+32 Phase3D/E tiles). Those are prior-agent claims, not re-verified here.
- Script-based checks (SHA, CSV membership, pixel metrics) are not visual inspection and are not presented as such.

## 10. Selected next step

**Do not train. Do not use the current split.** The next phase should build a small, verified, source-grouped pool cohort with completed person reannotation, then rebuild the split from READY-only rows. See `NEXT_PHASE_DECISION.md` and `NEXT_PHASE_EXECUTION_PLAN.md`.
