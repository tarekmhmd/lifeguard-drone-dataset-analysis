# Phase H — Final Dataset Recovery & Readiness Assessment Report
## `DrowningDetectionTracking_v1` | Autonomous AI Lifeguard Drone (Swimming Pools Only)

**Report Date:** 2026-11-03  
**Final Readiness Verdict:** **`UNAPPROVED FOR TRAINING`**  
**Scorecard Deliverable:** [`DATASET_READINESS_SCORECARD.csv`](file:///E:/lifeguardDron/DrowningDetection_v1_cleaning/04_recovery_investigation/DATASET_READINESS_SCORECARD.csv) (`22` rows)

---

> [!CAUTION]
> **MANDATORY TRAINING GATE VERDICT: `UNAPPROVED FOR TRAINING`**  
> `DrowningDetectionTracking_v1` **MUST NOT** be used to train or evaluate any model for the Autonomous AI Lifeguard Drone in its current state. Even after deduplicating the `14,262` byte-identical image copies (`60.47%` of the dataset) and excluding the out-of-scope and destructively augmented images, **`82.10%` (`3,463 / 4,218`) of the recoverable unique swimming-pool images require label remapping or manual re-annotation** due to the systematic `+4` / `(0,3) ↔ (4,5)` class-index offset corruption (`Class 4 = Out of Water` boxing drowning victims in water, and `Class 5 = drowning` boxing normal bystander swimmers) and opaque numeric class IDs (`0, 1, 2, 3`).

---

## 1. Direct Traceable Answers to All 11 Primary Investigation Questions (Section 2)

1. **Which images genuinely depict swimming-pool environments?**
   - **Verified Visually (`reviewer_status` in `VERIFIED_VISUAL` / `VERIFIED_SHA256_CLONE`):** **`383` files** (`146` directly visually inspected unique files + `237` byte-identical `SHA-256` clones) are visually confirmed clean swimming-pool scenes (`POOL`).
   - **Full-Dataset Candidate Total (`SCENE_CLASSIFICATION_FULL.csv`):** **`12,681` raw files (`53.76%`)**, representing **`5,107` unique `image_sha256` images (`54.77%` of unique SHAs)** across `SF02_output_NNNN`, `SF03_imageNNNN`, `SF06_kakaotalk`, `SF08_drowningN_mp4`, `SF10_video_2022_12_07`, `SF12_clipchamp_mp4`, `SF04_pure_digits` (`1..200_jpg` & `130000_jpg`), pool subsets of `SF13_stock_named`/`SF15`/`SF18`, and non-mosaic `SF01_youtube_N`. Excluding `SF01_youtube_N` (which requires frame-by-frame filtering of TV broadcast graphics/PiP), there are **`4,218` recoverable unique `image_sha256` swimming-pool images**.
2. **Which images contain usable human/swimmer annotations?**
   - **Verified Usable As-Is (`KEEP_CANDIDATE`):** Only **`18` unique visually verified swimming-pool images** (`52` raw files incl. exact duplicates) have verified clean semantic `5 (drowning)` / `6 (swimming)` bounding boxes ready without change.
   - **Candidate Usable Pending Quick Visual Check (`KEEP_IMAGE_REVIEW_LABEL`):** **`737` unique `image_sha256` pool images** (`2,107` raw files) in `output1..75` (`Class 5 = drowning`), `1..200_jpg` (`Class 6 = swimming`), clean `-Clipchamp-_mp4` (`Class 6`), and static pool photos carry semantic `5`/`6` boxes without class-offset collision, requiring only rapid visual confirmation (`755` total unique pool images with usable `5`/`6` box semantics).
   - **Good Box Geometry, Corrupted Class ID (`KEEP_IMAGE_REANNOTATE`):** **`3,463` unique `image_sha256` pool images** have usable human/swimmer box coordinates (where not rotated by patch augmentation) whose class IDs must be remapped or re-annotated.
3. **Which images contain incorrect or destructive augmentations?**
   - **`6,037` raw files (`25.60%`) = `2,146` unique `image_sha256` images (`23.02%`)** are ruined by destructive pre-baked Roboflow augmentations (`EXCLUDE_DESTRUCTIVE_AUGMENTATION`):
     - **`4,894` raw files (`1,691` unique SHAs in `CORRUPTED` + `7` additional cutout SHAs = `1,698` unique SHAs)** have **synthetic salt-and-pepper speckle or solid black/gray occlusion rectangles** stamped directly over the swimmers inside the annotated bounding boxes (`n_cutout_boxes > 0`).
     - **`1,143` raw files (`448` unique SHAs)** are **pre-baked 2×2 Mosaic composites** (`SUSPECTED_MOSAIC`, predominantly in `SF01_youtube_N`) with hard seams near `x=320, y=320` and seam-truncated boxes.
     - Additionally, hundreds of frames in `SF08_drowningN_mp4`, `SF04_pure_digits` (`1..200_jpg`), and `SF13_stock_named` exhibit **localized bounding-box patch rotation/shear** (where the rectangular region inside the bounding box was rotated by `±15°` relative to the surrounding water).
4. **Which labels have reliable semantics?**
   - **None of the 7 classes (`0..6`) can be trusted globally across the raw dataset.**
   - **Within isolated source families:**
     - `Class 5 (drowning)` is reliable **only** in `output1..75` (`41` unique clean SHAs, `153` raw files).
     - `Class 6 (swimming)` is reliable **only** in `1..200_jpg` (`587` unique clean SHAs) and clean frames of `-Clipchamp-_mp4` (`41` unique clean SHAs), after excluding `youtube-N` (where `Class 6` boxes the TV news PiP sign-language interpreter!) and `white-water_mp4` (river tubing).
5. **Which source groups contain incompatible annotation schemas?**
   - **Numeric Class Groups (`0, 1, 2, 3` = `'3', '4', '5', '6'`):** `output_0001..0600` (`0,2,3`), `SF06_kakaotalk` (`0,2,3`, where `2` = floating child in water vs. deck bystander in `output_NNNN`), `SF05_visdrone_0000` (`1` = street pedestrian), `130319..130473_jpg` (`1;3`), and `youtube-N` (`0,1,3,4,5,6`).
   - **Systematic `+4` and `(0,3) ↔ (4,5)` Class-Offset Inverted Groups (`602` source prefixes / `195` byte-identical `image_sha256` groups):** `SF08_drowningN_mp4` (`drowning1/47/20/50..55`), `SF10_video_2022_12_07`, `Untitled-video-Made-with-Clipchamp-1-_mp4`, `SF03_imageNNNN` (`THAIRATH TV 32`), `SF13_stock_named`, `SF11_videoN_mp4`, `SF14_val_train_test_N`, `SF15_img_pic_photo`, and `SF18_other_static`. In all of these groups, **`Class 4 (Out of Water)` boxes drowning victims submerged in water**, and in 2-class videos **`Class 5 (drowning)` boxes the normal bystander swimmers**!
   - **Segmentation Polygons in a Detection Dataset:** `15` files in `3-727-...-iStock_mp4-2_jpg` (`6` files) and `mp4-3_jpg` (`9` files).
6. **Which images are exact duplicates?**
   - Cataloged exhaustively in [`FULL_DUPLICATE_GROUPS.csv`](file:///E:/lifeguardDron/DrowningDetection_v1_cleaning/04_recovery_investigation/FULL_DUPLICATE_GROUPS.csv): **`7,317` multi-file `image_sha256` groups** encompass **`21,579` files (`91.49%` of raw dataset)**, meaning **`14,262` files (`60.47%`) are redundant byte-identical copies** (`6,413` triplets from Roboflow's 3× export, `725` pairs, `79` quadruplets, `26` quintuplets, `74` sextuplets). Only **`9,324` unique `image_sha256` images** exist in `DrowningDetectionTracking_v1`.
7. **Which images are near-duplicates or consecutive frames from the same video?**
   - Cataloged in `phash_pairs_le12.csv` and [`FULL_LEAKAGE_RELATIONSHIPS.csv`](file:///E:/lifeguardDron/DrowningDetection_v1_cleaning/04_recovery_investigation/FULL_LEAKAGE_RELATIONSHIPS.csv): **`78,680` pairs** of distinct `image_sha256` groups have `min(phash_dist, phash_flip_dist) <= 6` bits (including **`9,641` pairs at distance `0`** from horizontal flips, exposure shifts, and re-encoded duplicates). Furthermore, `output_0001..0600` (`2,276` raw / `892` unique SHAs), `image1..3763` (`1,916` raw / `734` unique SHAs), and `drowning1/47/50..55_mp4` (`2,265` raw / `999` unique SHAs) are each **single continuous videos** split across hundreds of prefixes.
8. **Which source groups cross `train`/`valid`/`test` boundaries?**
   - **`99` exact `image_sha256` groups (`365` files)**, **`685` `.rf.` `source_prefix` groups (`12,667` files, `53.71%`)**, and **`431` `atomic_video_unit` groups (`19,645` files, `83.29%`)** cross `train`/`valid`/`test` boundaries — contaminating **`96.87%` of `valid`** and **`95.91%` of `test`**.
9. **Which annotations can potentially be retained?**
   - **`755` unique `image_sha256` pool images** (`18` `KEEP_CANDIDATE` visually verified + `737` `KEEP_IMAGE_REVIEW_LABEL` pending quick visual check, primarily in `output1..75` [`Class 5`], `1..200_jpg` [`Class 6`], and clean `-Clipchamp-_mp4` [`Class 6`]).
   - In addition, deterministic script-based class-ID remapping (followed by human verification) can recover the bounding box coordinates on **`output_0001..0600` (`697` unique clean SHAs: map `0,3 -> swimming`, `2 -> out_of_water`)** and **`SF06_kakaotalk` (`175` unique clean SHAs: map `0,2,3 -> swimming`)**.
10. **Which images require manual re-annotation?**
    - **`3,463` unique `image_sha256` swimming-pool images (`KEEP_IMAGE_REANNOTATE`)** require class remapping or manual box/class correction (especially `SF03_imageNNNN` where poolside bystanders are unannotated and `Class 4` marks the drowning child, and `SF08_drowningN_mp4` / `SF12_clipchamp_mp4` / `SF10_video_2022_12_07` where `(0,3) ↔ (4,5)` class-offset inversion and box-patch rotations occur), plus **`1,813` `HUMAN_REVIEW_REQUIRED` unique SHAs** (`SF01_youtube_N` and `UNCERTAIN` crops).
11. **Can a reliable pool-only dataset be constructed from the available material?**
    - **CONDITIONAL YES — But Only After Deduplication, Frame Capping, and Re-Annotation/Verification:**
      - From the `23,586` raw files (`9,324` unique SHAs), **`4,218` unique `image_sha256` clean swimming-pool images** (`1,935` after capping giant single videos at `<=150` train / `<=50` eval frames in [`PROPOSED_SPLIT_MANIFEST.csv`](file:///E:/lifeguardDron/DrowningDetection_v1_cleaning/04_recovery_investigation/PROPOSED_SPLIT_MANIFEST.csv)) can be recovered with **zero cross-split leakage**.
      - However, because **3 videos account for `53.30%` (`2,248 / 4,218`) of the recoverable unique pool images** and only `755` unique pool images have uncorrupted semantic class IDs (`5`/`6`), constructing a production-grade dataset for the Autonomous AI Lifeguard Drone requires **(a) human visual review of the `737` `KEEP_IMAGE_REVIEW_LABEL` images, (b) re-annotation/remapping of the `1,180` capped `KEEP_IMAGE_REANNOTATE` pool images, and (c) supplementing with additional independent overhead drone pool recordings**.

---

## 2. Complete Dataset Readiness Scorecard (`DATASET_READINESS_SCORECARD.csv`)

| # | Metric Name | Value | `% / Ratio` | Evidence Basis |
| :---: | :--- | :---: | :---: | :--- |
| **1** | `total_images_inspected_inventory` | **23,586** | `100.00%` | `VERIFIED_EXACT` (`train=21,369`, `valid=1,215`, `test=1,002`) |
| **1b** | `direct_visual_inspection_unique_files` | **279** | `1.18%` | `VERIFIED_VISUAL` (`106` Phase 1 rows + `183` Phase 2 files = `279` net unique) |
| **1c** | `direct_visual_plus_exact_sha256_clones` | **691** | `2.93%` | `VERIFIED_VISUAL_AND_SHA_CLONE` (`279` direct + `412` exact SHA-256 clones) |
| **2** | `pending_visual_review_images` | **22,895** | `97.07%` | `VERIFIED_EXACT` (`reviewer_status = PENDING_REVIEW` in `SCENE_CLASSIFICATION_FULL.csv`) |
| **3** | `confirmed_pool_images_visual_and_sha_clones` | **383** | `1.62%` | `VERIFIED_VISUAL` (`146` direct + `237` exact SHA-256 clones) |
| **3b** | `candidate_pool_images_total_raw` | **12,681** | `53.76%` | `CANDIDATE_GROUP_PROPAGATED` (`5,107` unique `image_sha256` images) |
| **4** | `out_of_scope_images_total_raw` | **2,643** | `11.21%` | `VERIFIED_AND_PROPAGATED` (`1,147` unique `image_sha256` images) |
| **5** | `uncertain_images_total_raw` | **2,225** | `9.43%` | `VERIFIED_AND_PROPAGATED` (`931` unique `image_sha256` images) |
| **6** | `corrupted_or_mosaic_images_total_raw` | **6,037** | `25.60%` | `VERIFIED_PIXEL_FORENSICS` (`4,894` noise-cutout + `1,143` 2×2 Mosaic = `2,139` unique SHAs) |
| **7** | `exact_duplicate_redundant_images` | **14,262** | `60.47%` | `VERIFIED_EXACT_SHA256` (`7,317` multi-file groups; only `9,324` unique SHAs exist) |
| **8** | `near_duplicate_unique_sha_pairs_le6` | **78,680** | `N/A (pairs)` | `VERIFIED_PHASH_DHASH` (`9,641` pairs at `min_dist == 0`) |
| **9** | `cross_split_leakage_source_prefixes` | **685** | `25.58%` prefixes (`53.71%` imgs) | `VERIFIED_EXACT` (`12,667` imgs in cross-split prefixes; `19,645` [`83.29%`] in cross-split `atomic_video_unit`s) |
| **10** | `usable_annotations_without_change_verified` | **18** | `0.08%` (`0.19%` unique SHA) | `VERIFIED_VISUAL` (`KEEP_CANDIDATE`) |
| **10b** | `usable_annotations_pending_quick_visual_check` | **737** | `3.12%` (`7.90%` unique SHA) | `CANDIDATE_SEMANTIC_POOL` (`KEEP_IMAGE_REVIEW_LABEL`) |
| **11** | `recoverable_pool_images_needing_reannotation` | **3,463** | `14.68%` (`37.14%` unique SHA) | `CANDIDATE_REANNOTATE` (`KEEP_IMAGE_REANNOTATE`) |
| **12** | `images_requiring_human_review` | **1,813** | `7.69%` (`19.44%` unique SHA) | `CANDIDATE_HUMAN_REVIEW` (`HUMAN_REVIEW_REQUIRED`) |
| **13** | `excluded_images_total` | **17,555** | `74.43%` | `VERIFIED_EXCLUSION` (`14,262` exact dup + `2,146` destructive aug + `1,094` out-of-scope + `53` wrong object) |
| **14** | `estimated_final_clean_pool_dataset_size` | **4,218** (`1,935` capped) | `17.88%` raw (`45.24%` unique SHA) | `RECOVERABLE_UNIQUE_POOL_IMAGES` (`2,890 / 664 / 664` full; `941 / 524 / 470` frame-capped) |
| **15** | `estimated_clean_drowning_class_coverage` | **2,962** | `N/A` | `RECOVERABLE_POOL_WITH_DROWNING` (`41` clean `output1..75` + `~2,921` needing offset/numeric remap) |
| **16** | `estimated_clean_swimming_class_coverage` | **2,079** | `N/A` | `RECOVERABLE_POOL_WITH_SWIMMERS` (`628` clean `Class 6` + `~1,451` needing numeric/offset remap) |
| **17** | `estimated_clean_out_of_water_class_coverage` | **159** | `N/A` | `REQUIRES_REANNOTATION` (`>90%` of existing `Class 4` boxes are `+4`-shifted drowning victims in water!) |
| **18** | `overall_dataset_readiness_rating` | **`UNAPPROVED_FOR_TRAINING`** | **`0% Ready As-Is`** | **`MANDATORY_GATE_BLOCKED`** |

