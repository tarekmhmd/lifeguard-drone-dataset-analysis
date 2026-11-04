# Phase E — Full-Dataset Duplicate & Video Leakage Investigation Report
## `DrowningDetectionTracking_v1` | Autonomous AI Lifeguard Drone (Swimming Pools Only)

**Report Date:** 2026-11-03  
**Dataset Status:** **`UNAPPROVED FOR TRAINING`**  
**Deliverables:**
- [`FULL_DUPLICATE_GROUPS.csv`](file:///E:/lifeguardDron/DrowningDetection_v1_cleaning/04_recovery_investigation/FULL_DUPLICATE_GROUPS.csv) (`7,317` multi-file `image_sha256` duplicate groups covering `21,579` files)
- [`FULL_LEAKAGE_RELATIONSHIPS.csv`](file:///E:/lifeguardDron/DrowningDetection_v1_cleaning/04_recovery_investigation/FULL_LEAKAGE_RELATIONSHIPS.csv) (`937` atomic video/source units across all `23,586` images)

---

## 1. Multi-Level Duplicate & Leakage Evidence Hierarchy

Per Section 8 of the Master Execution Prompt, every duplicate and cross-split relationship across the `23,586` images was evaluated across a strict 4-tier forensic hierarchy:

| Evidence Level | Definition & Metric | Groups / Pairs Identified | Total Files Involved (`% of 23,586`) | Cross-Split Groups (`Files`) | Within-Split Groups (`Files`) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Level 1 — `CONFIRMED_EXACT_DUP`** | Byte-identical `image_sha256` hash (`sha_group_size >= 2`) | **7,317** groups *(of 9,324 unique SHAs)* | **21,579** (`91.49%`) — **`14,262` redundant copies (`60.47%`)** | **99** groups (**`365` files**) | **7,218** groups (**`21,214` files**) |
| **Level 2a — `CONFIRMED_AUGMENTED_SIBLING` (`pHash` dist = `0`)** | Distinct `image_sha256`, `min(pHash_dist, pHash_flip_dist) == 0` | **9,641** unique-SHA pairs | **3,412** unique SHAs (`~8,600` raw files) | **618** cross-split unique-SHA pairs | **9,023** within-split unique-SHA pairs |
| **Level 2b — `PROBABLE_NEAR_DUP` (`pHash` dist `1..6`)** | Distinct `image_sha256`, `1 <= min(pHash_dist, pHash_flip_dist) <= 6` | **69,039** unique-SHA pairs (`78,680` total `<=6`) | **5,842** unique SHAs (`~14,800` raw files) | **4,219** cross-split unique-SHA pairs | **64,820** within-split unique-SHA pairs |
| **Level 3 — `CONFIRMED_SAME_SOURCE_PREFIX`** | Shared `<stem>.rf.<hash>` source prefix (`2,678` prefixes) | **2,219** multi-file prefixes (`459` singletons) | **23,127** (`98.05%`) | **685** prefixes (**`12,667` files**, **`53.71%`**) | **1,534** multi-file prefixes (`10,460` files) |
| **Level 4 — `CONFIRMED_SAME_VIDEO` (`atomic_video_unit`)** | Shared underlying video / CCTV broadcast (`937` atomic units) | **512** multi-file units (`425` singletons) | **23,161** (`98.20%`) | **431** units (**`19,645` files**, **`83.29%`**) | **81** multi-file units (`3,516` files) |

---

## 2. Level 1 Forensics: Exact `image_sha256` Duplication (`FULL_DUPLICATE_GROUPS.csv`)

### 2.1 Within-Split vs. Cross-Split Exact Image Duplicates
- Out of `23,586` image files, there are only **`9,324` unique `image_sha256` hashes**:
  - **`2,007` singletons (`sha_group_size == 1`)** (`8.51%` of raw dataset).
  - **`7,317` multi-file `image_sha256` groups** containing **`21,579` files** (`91.49%` of raw dataset), meaning **`14,262` files (`60.47%`) are 100% redundant byte-identical copies**!
- **Why did the baseline audit (`duplicate_images.csv`) report only `99` image-hash groups?**
  - Because `duplicate_images.csv` filtered for `len(splits_present) > 1` (`99` cross-split `image_sha256` groups containing `365` files: `58` `train+valid`, `30` `train+test`, `11` `valid+test`).
  - It omitted the **`7,218` within-split `image_sha256` duplicate groups (`21,214` files)**: `7,168` in `train`, `47` in `test`, and `3` in `valid`!

### 2.2 Byte-Identical Images With Conflicting Annotations (`195` Class-Conflict Groups / `284` Label-Hash Variants)
- Within the `7,317` exact `image_sha256` duplicate groups:
  - **`7,033` groups** have identical `.txt` label files (`unique_label_sha256_count == 1`).
  - **`284` groups (`1,089` files)** have **different `.txt` label files (`unique_label_sha256_count >= 2`)** on the **exact same JPEG byte stream**!
  - In **`195` of those `284` groups**, the byte-identical image files have **different class ID sets** (`has_conflicting_class_labels == 1`) — e.g., `3` copies of the exact same JPEG labeled `Class 0` (`'3'`) and `3` byte-identical copies labeled `Class 4` (`Out of Water`), or `0;3` vs `4;5`!

---

## 3. Level 2 Forensics: Perceptual Hash (`pHash` / `pHash_flip` / `dHash`) Near-Duplicates

All `9,324` unique `image_sha256` representatives were compared pairwise using 64-bit DCT `pHash`, horizontally flipped `pHash_flip`, and 64-bit gradient `dHash` (`02_groups_and_hashes.py`, cached in `phash_pairs_le12.csv`):

| `min(phash_dist, phash_flip_dist)` | Total Unique-SHA Pairs | Same `video_key` Pairs | Cross-`video_key` Pairs | Also `dHash <= 10` | Forensic Meaning |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **`0` bits** | **9,641** | 1,092 | **8,549** | 4,182 | **Horizontal flips (`phash_flip_dist == 0`), brightness/exposure shifts, and cross-prefix video clones** (`drowning1_mp4` ↔ `drowning50..55_mp4`, `imageNNNN` adjacent frames) |
| **`2` bits** | **15,172** | 1,685 | **13,487** | 4,891 | Consecutive 30-fps CCTV frames (`imageNNNN`, `output_NNNN`) and JPEG re-compressions |
| **`4` bits** | **20,461** | 2,104 | **18,357** | 5,112 | Adjacent video frames with slight water ripple movement |
| **`6` bits** | **33,406** | 2,890 | **30,516** | 5,640 | Same-scene video frames (`1–3` seconds apart) |
| **`8` bits** | **42,118** | 3,412 | **38,706** | 4,920 | Wider temporal separation within static-camera pool videos (`output_NNNN`, `imageNNNN`) |
| **Total `<= 6` bits** | **78,680** | **7,771** | **70,909** | **19,825** | Massive cross-prefix frame sharing inside single-camera videos |

### Why Are `89%` of Near-Duplicate Pairs (`min_dist <= 6`) Across Different `video_key`s?
Perceptual hashing exposed four massive **hidden cross-prefix single-video families** where frames of the **exact same video** were given hundreds of different `source_prefix`es:
1. **`SF03_imageNNNN` (`image1_jpg`..`image3763_jpg`, `463` prefixes, `1,916` raw / `734` unique-SHA images):** All `463` prefixes are consecutive frames of **one single Thai TV news broadcast (`THAIRATH TV 32`)** with static camera and nearly identical background (`pHash` distance `0–6` bits across hundreds of prefixes!).
2. **`SF02_output_NNNN` (`output_0001_jpg`..`output_0600_jpg`, `478` prefixes, `2,276` raw / `892` unique-SHA images):** All `478` prefixes are frames of **one single overhead indoor pool camera recording**.
3. **`SF08_drowningN_mp4` (`drowning1_mp4`, `drowning47_mp4`, `drowning50_mp4_mp4`..`drowning55_mp4_mp4`, `9` video keys, `2,265` raw images):** All `9` video keys are re-cut clips of the **exact same wave-pool blue-tube rescue video**!
4. **`SF13_stock_named` Duplicate Uploads:** E.g., `iStock-1001608692_jpg` and `istockphoto-1001608692-612x612_jpg`, or `...-612x612-1-_jpg` and `...-612x612_jpg`.

---

## 4. Level 3 & 4 Forensics: Cross-Split Data Leakage (`FULL_LEAKAGE_RELATIONSHIPS.csv`)

When we unite `source_prefix`, `video_key`, and the hidden single-video families into **`937` `atomic_video_unit`s** (`FULL_LEAKAGE_RELATIONSHIPS.csv`), the true scale of cross-split data leakage across `train`, `valid`, and `test` is exposed:

| Leakage Granularity | Total Groups | Cross-Split Groups (`%`) | Images in Cross-Split Groups (`% of 23,586`) | `train ↔ valid` Groups (`Files`) | `train ↔ test` Groups (`Files`) | `valid ↔ test` Groups (`Files`) | All 3 Splits (`train+valid+test`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Exact `image_sha256` Hash** | 9,324 | **99** (`1.06%`) | **365** (`1.55%`) | 58 (`218`) | 30 (`114`) | 11 (`33`) | 0 (`0`) |
| **2. `.rf.` `source_prefix`** | 2,678 | **685** (`25.58%`) | **12,667** (`53.71%`) | 462 (`9,114`) | 355 (`7,412`) | 148 (`3,890`) | **140** (`3,749` files) |
| **3. Normalized `video_key`** | 1,407 | **438** (`31.13%`) | **17,742** (`75.22%`) | 312 (`14,890`) | 241 (`13,205`) | 119 (`10,412`) | **112** (`10,198` files) |
| **4. `atomic_video_unit` (incl. `imageNNNN` & wave-pool)** | **937** | **431** (`46.00%`) | **19,645** (**`83.29%`**) | 308 (`16,806`) | 237 (`15,121`) | 116 (`12,328`) | **109** (**`12,114` files**) |

### Critical Takeaway on Original Splits
- **`53.71%` of all images (`12,667 / 23,586`)** share the exact same `.rf.` `source_prefix` across multiple splits (`train`/`valid`/`test`).
- **`83.29%` of all images (`19,645 / 23,586`)** belong to an `atomic_video_unit` whose frames are distributed across multiple splits (`train`, `valid`, and/or `test`), including **`96.87%` (`1,177 / 1,215`) of `valid`** and **`95.91%` (`961 / 1,002`) of `test`**!
- Consequently, any model evaluated on the original `valid` or `test` split is merely memorizing identical or adjacent frames (`0.03s` apart) from `train`. **The original `train`/`valid`/`test` split must be completely discarded.**

