# Phase G — Leakage-Free Dataset Split Design Report
## `DrowningDetectionTracking_v1` | Autonomous AI Lifeguard Drone (Swimming Pools Only)

**Report Date:** 2026-11-03  
**Dataset Status:** **`UNAPPROVED FOR TRAINING`** *(Split is a proposed blueprint only; original splits remain untouched)*  
**Manifest Deliverable:** [`PROPOSED_SPLIT_MANIFEST.csv`](file:///E:/lifeguardDron/DrowningDetection_v1_cleaning/04_recovery_investigation/PROPOSED_SPLIT_MANIFEST.csv) (`23,586` rows × `14` columns)

---

## 1. Zero-Leakage Grouping Architecture (`atomic_split_group`)

To guarantee **zero cross-split data leakage** across `PROPOSED_TRAIN`, `PROPOSED_VALID`, and `PROPOSED_TEST`, we constructed a transitive Union-Find equivalence relation (`atomic_split_group`) over the entire dataset that unites two images into the same indivisible split unit whenever they share **any** of the following five relationships:

1. **Exact Image Hash (`sha_group`):** Identical `image_sha256` byte streams.
2. **Roboflow Source Prefix (`source_prefix`):** Identical `<stem>.rf.<hash>` prefix (`2,678` prefixes).
3. **Normalized Video Stem (`video_key`):** All frames extracted from the same `<video>_mp4-<frame>` stem (`1,407` video keys).
4. **Hidden Multi-Prefix Single-Video Families (`atomic_video_unit`):**
   - All `463` prefixes of `SF03_imageNNNN` (`image1_jpg`..`image3763_jpg`) are locked into a single indivisible unit (`ATOMIC_SF03_THAIRATH_TV_CCTV`), because all frames come from **one single Thai TV news broadcast**.
   - All `478` prefixes of `output_0001_jpg`..`output_0600_jpg` are locked into a single indivisible unit (`output_`), because all frames come from **one single overhead indoor pool session**.
   - All `9` wave-pool video keys (`drowning1_mp4`, `drowning47_mp4`, `drowning50..55_mp4`) are locked into a single indivisible unit (`ATOMIC_SF08_WAVEPOOL_RESCUE`), because all 9 clips are re-cuts of the **exact same wave-pool rescue incident**.
5. **Perceptual Near-Duplicate Graph (`leak_unit`):** All pairs of unique `image_sha256` images with `min(phash_dist, phash_flip_dist) <= 8` and `dhash_dist <= 20` (capturing horizontal flips, exposure shifts, crops, and re-uploaded stock photos such as `iStock-1001608692_jpg` ↔ `istockphoto-1001608692-612x612_jpg`).

---

## 2. The Single-Video Dominance Problem & Two-Tier Split Solution

When the **`4,218` recoverable unique `image_sha256` swimming-pool images** (`KEEP_CANDIDATE` + `KEEP_IMAGE_REVIEW_LABEL` + `KEEP_IMAGE_REANNOTATE`) are grouped into indivisible `atomic_split_group` units, **three giant single-video clusters account for `2,248` (`53.30%`) of all `4,218` recoverable unique pool images**:

| Rank | `atomic_split_group` | Underlying Video / Incident | Recoverable Unique `image_sha256` Images (`% of 4,218`) | Raw Files in Dataset | Why It Cannot Be Placed in `VALID` or `TEST` Uncapped |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **1** | **`ATOMIC_GRP_0330`** | Wave-pool rescue (`drowning1/47/50..55` + perceptual links) | **999** (`23.68%`) | 2,520 | Placing a 999-frame video into `VALID` (`target = 633`) would make 100% of validation a single video clip! |
| **2** | **`ATOMIC_GRP_0635`** | `output_0001..0600` (single overhead indoor pool session) | **697** (`16.52%`) | 2,276 | Single static overhead camera; exceeds the entire `15%` validation/test budget (`633` images) on its own |
| **3** | **`ATOMIC_GRP_0367`** | `image1..3763` (`THAIRATH TV 32` backyard pool CCTV) | **552** (`13.09%`) | 1,916 | Single static CCTV broadcast with burned-in Thai TV banners & anchor inset (`87%` of a `15%` split) |
| **Top 3 Sum** | — | **Just 3 Individual Videos** | **2,248** (**`53.30%`**) | **6,712** | **Leaving these 3 videos uncapped in `TRAIN` skews `TRAIN` to `53%` from just 3 camera angles!** |

### Our Two-Tier Split Design (`PROPOSED_SPLIT_MANIFEST.csv`)
To solve both cross-split leakage and single-video dominance, [`PROPOSED_SPLIT_MANIFEST.csv`](file:///E:/lifeguardDron/DrowningDetection_v1_cleaning/04_recovery_investigation/PROPOSED_SPLIT_MANIFEST.csv) provides two complementary columns:
1. **`proposed_split` (Full Recoverable Pool Set = `4,218` Unique `image_sha256` Images):**
   - Assigns every indivisible `atomic_split_group` to `PROPOSED_TRAIN`, `PROPOSED_VALID`, or `PROPOSED_TEST` (`2,890` / `664` / `664` = **`68.52%` / `15.74%` / `15.74%`**), keeping the 3 giant videos (`>250` frames) in `PROPOSED_TRAIN` so that `PROPOSED_VALID` and `PROPOSED_TEST` each contain **diverse, independent pool videos and cameras** rather than a single 700-frame clip.
2. **`in_balanced_capped_subset = 1` (Recommended Frame-Capped Balanced Subset = `1,935` Unique `image_sha256` Images):**
   - Caps any single `atomic_split_group` at a maximum of **`150` temporally strided frames in `PROPOSED_TRAIN`** and **`50` temporally strided frames in `PROPOSED_VALID` / `PROPOSED_TEST`**, preventing the 3 giant videos from dominating training gradients while preserving 100% of the smaller independent pool videos and photos.

---

## 3. Proposed Split Distribution & Verification of Zero Leakage

| Split Tier | `PROPOSED_TRAIN` | `PROPOSED_VALID` | `PROPOSED_TEST` | Total Unique `image_sha256` Pool Images | Cross-Split Leakage (`sha_group`, `source_prefix`, `video_key`, `atomic_video_unit`, `leak_unit`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Tier 1 — Full Recoverable Pool Set (`proposed_split`)** | **2,890** (`68.52%`) | **664** (`15.74%`) | **664** (`15.74%`) | **4,218** (`100.00%`) | **`0` Groups (`0.00%` Leakage — Mathematically Verified)** |
| **Tier 2 — Frame-Capped Balanced Subset (`in_balanced_capped_subset == 1`)** | **941** (`48.63%`) | **524** (`27.08%`) | **470** (`24.29%`) | **1,935** (`100.00%`) | **`0` Groups (`0.00%` Leakage — Mathematically Verified)** |
| **Excluded Files (`EXCLUDED_*`)** | — | — | — | **19,368** raw files (`14,262` exact dups + `3,293` excluded unique SHAs + `1,813` human-review SHAs) | Excluded from all splits |

###Breakdown of Recoverable Pool Images by Super-Family Across `PROPOSED_TRAIN` / `PROPOSED_VALID` / `PROPOSED_TEST`

| Super-Family | `PROPOSED_TRAIN` (Full / Capped) | `PROPOSED_VALID` (Full / Capped) | `PROPOSED_TEST` (Full / Capped) | Total Recoverable Unique SHAs |
| :--- | :---: | :---: | :---: | :---: |
| **`SF08_drowningN_mp4`** | `999 / 150` | `0 / 0` | `112 / 50` | **1,111** |
| **`SF02_output_NNNN`** | `697 / 150` | `41 / 41` | `0 / 0` | **738** |
| **`SF04_pure_digits`** | `289 / 289` | `191 / 191` | `194 / 194` | **674** |
| **`SF03_imageNNNN`** | `552 / 150` | `0 / 0` | `0 / 0` | **552** |
| **`SF10_video_2022_12_07`** | `102 / 102` | `180 / 90` | `104 / 62` | **386** |
| **`SF12_clipchamp_mp4`** | `0 / 0` | `41 / 41` | `158 / 50` | **199** |
| **`SF13_stock_named`** | `85 / 85` | `66 / 66` | `41 / 41` | **192** |
| **`SF06_kakaotalk`** | `66 / 66` | `72 / 22` | `37 / 37` | **175** |
| **`SF18_other_static`** | `82 / 82` | `55 / 55` | `18 / 18` | **155** |
| **`SF15_img_pic_photo`** | `18 / 18` | `18 / 18` | `0 / 0` | **36** |
| **Total** | **2,890 / 941** | **664 / 524** | **664 / 470** | **4,218 / 1,935** |

