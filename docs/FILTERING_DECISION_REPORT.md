# Filtering Decision Report — DrowningDetectionTracking_v1

Date: 2026-11-02
Scope: read-only audit + filtering decision. No dataset file was modified.
Status labels used: **VERIFIED**, **PENDING**, **UNCERTAIN**, **BLOCKED**.

---

## 1. Current dataset status

| Item | Value | Status |
|---|---|---|
| Images / labels | 23,586 / 23,586 (train 21,369, valid 1,215, test 1,002) | VERIFIED |
| Valid YOLO boxes | 49,040 | VERIFIED |
| Image integrity | all decode; all 640×640; 0 corrupt / 0 zero-byte / 0 unsupported | VERIFIED |
| Image-label pairing | 0 missing, 0 orphan, 0 duplicate stems | VERIFIED |
| Malformed annotation lines | 15 (all train) | VERIFIED |
| Empty label files | 1 (test) | VERIFIED |
| Large boxes (>90% area) | 67 (train 60, valid 5, test 2) | VERIFIED |
| Class schema | 7 IDs (0–6), mixed numeric + semantic | VERIFIED |
| Scene classification | 3 of 23,586 classified | PENDING |
| Overall readiness for filtering | Not ready | BLOCKED |

## 2. Verified class mapping

Semantic meanings are derived only from the YAML `names`; the README provides no class definitions. The public Roboflow page documents three semantic classes, while the local export declares seven. This contradiction means the numeric classes cannot be interpreted.

| ID | Name | Boxes | Images | Semantic meaning | Confidence |
|---|---:|---:|---:|---|---|
| 0 | `3` | 9,890 | 8,597 | Numeric; undocumented | UNKNOWN |
| 1 | `4` | 4,683 | 370 | Numeric; undocumented | UNKNOWN |
| 2 | `5` | 1,432 | 920 | Numeric; undocumented | UNKNOWN |
| 3 | `6` | 9,729 | 4,737 | Numeric; undocumented | UNKNOWN |
| 4 | `Out of Water` | 8,160 | 7,750 | Person out of water (name only) | PARTIALLY VERIFIED |
| 5 | `drowning` | 4,409 | 2,650 | Drowning (name only; not visually verified) | PARTIALLY VERIFIED |
| 6 | `swimming` | 10,737 | 4,511 | Swimming (name only) | PARTIALLY VERIFIED |

The schema **cannot yet be safely interpreted** for the project because:

- Classes 0–3 have no documented meaning.
- The three semantic class names are not independently verified against image content.
- The dataset mixes two apparent schemas: numeric-only images (train 9,792; valid 570; test 436) and semantic-only images (train 10,968; valid 600; test 540), plus mixed (train 603; valid 45; test 25).

## 3. Scene classification progress

The derived manifest `scene_classification_manifest.csv` contains **3 classified images** out of 23,586:

| Index | Split | Category |
|---|---|---|
| 0 | train | POOL |
| 1 | train | OUT_OF_SCOPE |
| 2 | train | UNCERTAIN |

All three belong to the same source clip (`-Clipchamp-_mp4-10`). 23,583 images remain unclassified.

## 4. Leakage investigation results

| Finding | Groups | Files | Classification |
|---|---:|---:|---|
| Exact cross-split image hashes | 99 | 365 | **Confirmed leakage** (byte-identical) |
| Exact cross-split label hashes | 173 | 702 | **Confirmed duplication** (identical labels) |
| Same-source-prefix groups across splits | 685 | 12,667 | **Suspected temporal/source contamination** (not byte-identical) |
| Near-duplicate / perceptual | not computed | — | UNCERTAIN |
| Same filename across splits | 0 | 0 | VERIFIED (none) |

Evidence: many exact-label and exact-image groups pair `train` with `valid` and `test` (e.g., `drowning1_mp4-*`, `drowning20_mp4-*`, `10_MOV-*`, iStock stock clips). Same-source-prefix groups (e.g., `youtube-6`, `youtube-2`, `video40-*`, `iStock` clips) are distributed across splits.

**Remediation strategy (PROPOSED, not executed):**

1. Remove or deduplicate byte-identical images/labels across splits before any split assignment.
2. Rebuild splits at **source/video/recording-session level**, so all frames from one source stay in a single split.
3. Treat same-source-prefix groups as leakage candidates until verified; do not randomly distribute same-source frames.
4. Do **not** delete anything from the original dataset — only from a derived, documented copy.

## 5. Annotation quality results

### 5.1 Malformed lines (15, all train)

All 15 malformed lines:

- Begin with class ID 4.
- Contain 45, 47, 57, 59, or 61 fields.
- Are long sequences of normalized coordinate pairs (consistent with polygon/segmentation data, not a YOLO detection box).
- Occur in iStock drowning stock frames (`3-727-Drowning-Stock-Videos-and-Royalty-Free-Footage-iStock_mp4-2/3`).

Assessment:

- **Cannot be safely repaired** into a five-field bounding box: there is no unambiguous "this coordinate list = one box" semantics.
- 6 train label files contain only the malformed line and no valid box (VERIFIED from the prior line-level scan).
- Recommendation: **exclude** these 15 lines and the 6 only-malformed files from the derived dataset; record each in a repair log. Do not silently rewrite.

### 5.2 Large boxes (67)

60 class 0, 7 class 3 cover >90% of the image. These are likely full-frame/near-full-frame boxes; treat as **suspicious** until visually confirmed. Do not delete automatically.

### 5.3 Empty label (1, test)

`mvi_1627_vis_frame20...` — may be a legitimate negative or a missing annotation; visual review required.

### 5.4 Otherwise valid

No invalid class IDs, no invalid coordinates, no out-of-bounds boxes, no duplicate annotation lines, no tiny boxes.

## 6. Confirmed pool samples

3 (all from source clip `-Clipchamp-_mp4-10`): 1 POOL, 1 OUT_OF_SCOPE, 1 UNCERTAIN. This is too few to establish a pool subset.

## 7. Confirmed out-of-scope samples

1 (index 1). No large out-of-scope set has been confirmed yet.

## 8. Uncertain samples

1 (index 2) plus the entire unclassified population (23,583). All numeric-class semantics remain uncertain.

## 9. Recommended exclusions and reasons

| Item | Reason | Action |
|---|---|---|
| 15 malformed annotation lines | polygon-like, not YOLO detection | exclude lines from derived copy; log |
| 6 only-malformed label files | no valid annotation | exclude from derived copy; log |
| 67 large (>90%) boxes | likely full-frame / ambiguous | visual review; exclude if confirmed bogus |
| 1 empty test label | unresolved negative vs missing | visual review |
| Byte-identical cross-split duplicates | confirmed leakage | deduplicate in derived copy |
| Same-source cross-split frames | suspected temporal leakage | regroup by source, then re-split |
| Numeric-class images (0–3) | undocumented semantics | hold until classes resolved |

None of these are applied to the original dataset.

## 10. Remaining blockers

1. **Scene classification is essentially unstarted** (3/23,586). BLOCKED.
2. **Numeric class semantics 0–3 are unknown.** BLOCKED.
3. **Leakage remediation** must be designed and applied to a derived copy. PENDING.
4. **Semantic classes `drowning`/`Out of Water`/`swimming` are not visually verified.** PENDING.
5. **Original `data.yaml` is missing**; `data.fixed.yaml` has an invalid absolute `path`. PENDING.

## 11. Is the dataset ready for pool-only filtering?

**No. BLOCKED.**

The structural integrity is high (all images decode, pairing is complete, 49,040 valid boxes), but filtering requires scene classification and class-schema resolution, neither of which is complete.

## Recommendation (next action)

**B — Continue targeted scene classification.**

Rationale:

- The dataset is not structurally corrupt (all images readable, no pairing gaps, only 15 malformed lines / 0.03% of boxes).
- The blocking problems are **resolvable by inspection**, not fundamental: unknown numeric-class semantics, unfinished scene classification, and leakage that can be fixed in a derived copy.
- It is too early to **proceed with filtering** (A) because only 3 images are classified.
- It is too early to **stop using the dataset** (C) because the evidence does not show it is unusable — it shows it is unresolved.

Next concrete steps: (1) continue source-grouped, stratified scene classification in a derived manifest; (2) resolve numeric classes via visual review; (3) build a source-level split to remove confirmed/suspected leakage; (4) only then create a derived pool-only YOLO dataset.
