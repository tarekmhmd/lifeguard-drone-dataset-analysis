# VISION DUPLICATE AND LEAKAGE REPORT — `DrowningDetectionTracking_v1`

- **Investigation Date:** 2026-11-03
- **Agent Role:** Visual Inspection & Dataset Forensics Agent (`vision-agent`)
- **Dataset:** `DrowningDetectionTracking_v1` (`23,586` images across `train`/`valid`/`test`)
- **Dataset Approval Status:** **NOT APPROVED FOR TRAINING**

---

## 1. Executive Summary of Duplicate & Cross-Split Leakage Forensics

Direct cryptographic hashing (`SHA-256`) combined with pixel-level visual inspection confirms **severe, systemic cross-split data leakage** across `train`, `valid`, and `test` in `DrowningDetectionTracking_v1`:

| Leakage Classification | Dataset-Wide Audit Scope (`duplicate_images.csv` & `same_source_prefix.csv`) | Visually Inspected & Verified in This Session (`VERIFIED`) | Key Forensic Finding |
|---|---:|---:|---|
| **`CONFIRMED_EXACT_DUP`** | **`99` cross-split image-hash groups** (`365` byte-identical files across `train` and `valid`) | **`5` cross-split pairs (`10` images)** extracted, hashed, and visually inspected | Byte-for-byte identical JPEG files (identical `SHA-256` and byte size) appear simultaneously in `train` and `valid` under different `.rf.<hash>.jpg` filenames. |
| **`CONFIRMED_SAME_VIDEO`** | **`173` cross-split label-hash groups** (`702` files) + **`1,616` cross-split source-prefix groups** (out of `4,949` total source groups) | **`3` label-hash pairs (`6` images)** + **`5` multi-split source groups (`15` images)** + **cross-prefix `youtube-*` & `iStock` pairs** | Identical video frames or augmented siblings (rotated/flipped/blurred/cutout versions of the exact same frame) are distributed across `train`, `valid`, and `test`. Furthermore, identical videos span multiple different `youtube-*` prefixes (`youtube-2`, `youtube-5`, `youtube-6`). |
| **`SUSPECTED_SOURCE`** | **`12+` high-volume `youtube-*` prefix groups** (`youtube-0_jpg` through `youtube-11_jpg`, totaling `>2,800` images) | **`36` contact-sheet tiles** across `youtube-1_jpg` to `youtube-11_jpg` + `2` extra `youtube-*` tiles | `youtube-*` prefix groups are **NOT homogeneous single-video groups**: `31 / 36` inspected tiles are **2×2 Roboflow Mosaic composites** combining 4 unrelated source videos (pools, TV studio sign-language insets, dry grass, close-up water) into single images, while leaking the same constituent clips across `train`, `valid`, and `test`. |
| **`UNRESOLVED`** | Remaining uninspected groups among the `1,616` cross-split prefixes in `same_source_prefix.csv` | Listed for automated perceptual-hash / video-sequence grouping prior to any dataset rebuild | Cannot be trusted as independent splits without full base-frame deduplication and video-level re-splitting. |

---

## 2. Confirmed Exact Duplicates (`CONFIRMED_EXACT_DUP`)

### 2.1 Audit Scope (`99` Cross-Split Groups, `365` Files)
All `99` `exact_image_hash` groups reported in `duplicate_images.csv` (`365` total files) are **cross-split duplicates** (`train` vs `valid`). Typically, each group consists of `2`–`3` byte-identical copies in `train` and `1` byte-identical copy in `valid`, each assigned a different Roboflow `.rf.<hash>.jpg` suffix.

### 2.2 Visually Verified Cross-Split Exact Duplicate Pairs (`VERIFIED`)
We extracted, computed full `SHA-256` digests and byte sizes, and visually inspected the pixels of **5 cross-split pairs (`10` images)**:

| Pair # | Split A File (`train`) | Split B File (`valid`) | Additional Byte-Identical Copies in `train` | Verified `SHA-256` Digest & Size | Visual Scene & Evidence (`VERIFIED`) | Classification |
|---:|---|---|---|---|---|---|
| **1** | `train/images/youtube-6_jpg.rf.81eec4b1de0f1d1ead77a166aae7b901.jpg` | `valid/images/youtube-6_jpg.rf.0a32095866a2bdffa2a69beac7ce6d81.jpg` | `train/.../youtube-6_jpg.rf.6a87fbcf1dd4656dea3157ad19555917.jpg`, `train/.../youtube-6_jpg.rf.0fc42046b0629ccf8da830283ef0b401.jpg` | `cb6702763f83b66ffbd862851bbd7486732033e27fc8c0eafef108635fe15ba5` (`28,746` bytes) | Overhead view of indoor swimming pool with yellow-green coping (`POOL`); child in yellow shirt submerged in center (`cls=5`) and rescuer at bottom-right (`cls=6`). Pixels are 100% identical between `train` and `valid`. | **`CONFIRMED_EXACT_DUP`** |
| **2** | `train/images/youtube-5_jpg.rf.0e193c1be267b71d7e9c16a69b297789.jpg` | `valid/images/youtube-5_jpg.rf.4556a380e23a46965cf90a78073f1c21.jpg` | `train/.../youtube-5_jpg.rf.af206f3486bc580cbe62ed4f07819ccb.jpg`, `train/.../youtube-5_jpg.rf.7146de18baadef917ea94d27849ac320.jpg` | `eab5d12215aec5d5ea968ad9838a29635bf4b26c54471a9cca58f6b101f95226` (`26,120` bytes) | Adjacent video frame from the **exact same indoor yellow-green coping pool** as Pair 1 (`POOL`); child submerged in center and rescuer at bottom-right. Pixels are 100% identical between `train` and `valid`. | **`CONFIRMED_EXACT_DUP`** |
| **3** | `train/images/youtube-3_jpg.rf.ce3a53732b836d5f0d7e0392f6948e95.jpg` | `valid/images/youtube-3_jpg.rf.da35077ef926e54d2e4b4fd422a5801f.jpg` | `train/.../youtube-3_jpg.rf.d55f246feda41908b7c50f14f76b1710.jpg`, `train/.../youtube-3_jpg.rf.84a03ecfcd7fdd0734515cbc09534f1b.jpg` | `13dbd3edf4e4e0c5a05595b549b4d45aaff5e0b820654cb81febf90b2d8e6784` (`32,153` bytes) | Outdoor residential swimming pool with red brick coping and white lounge chair (`POOL`); person in red/black at far coping edge (`cls=4`). Pixels are 100% identical between `train` and `valid`. | **`CONFIRMED_EXACT_DUP`** |
| **4** | `train/images/youtube-4_jpg.rf.37bf12bc1d8864de7d4ec6167cae8260.jpg` | `valid/images/youtube-4_jpg.rf.d1091e6e57720a75e34e2e5ac1949fb3.jpg` | `train/.../youtube-4_jpg.rf.dbd805d2c98d93c32a525b0e02b571e3.jpg`, `train/.../youtube-4_jpg.rf.0d370d26b619042bb856f6876eb27de2.jpg` | `6b4f4497c671a5c79b6fa884bb07791cc9b4bcd6e85e06b75fa4e1778f5e7318` (`31,025` bytes) | Subsequent video frame from the **exact same outdoor red-coping residential pool** as Pair 3 (`POOL`), now showing the person splashing in the water (`cls=5`). Pixels are 100% identical between `train` and `valid`. | **`CONFIRMED_EXACT_DUP`** |
| **5** | `train/images/youtube-2_jpg.rf.5e4807e549176fb7d19e8f9b13f34313.jpg` | `valid/images/youtube-2_jpg.rf.52809fbb5735ce458f5a1a784ba6b9c7.jpg` | `train/.../youtube-2_jpg.rf.62c76f3a47c34744d2b183d24fdc89cd.jpg` | `923d0100485ee1b7e83ca992ff2889d02b9d3bc21d3ca5d3d29f539101021eb0` (`25,211` bytes) | Another video frame from the **exact same indoor yellow-green coping pool** as Pairs 1 & 2 (`POOL`), showing the child in yellow shirt and yellow/red lane rope at bottom-right. Pixels are 100% identical between `train` and `valid`. | **`CONFIRMED_EXACT_DUP`** |

---

## 3. Confirmed Same-Video & Augmented-Sibling Leakage (`CONFIRMED_SAME_VIDEO`)

### 3.1 Cross-Split Exact Label Hash Pairs (`173` Cross-Split Groups, `702` Files)
In `duplicate_images.csv`, `173` groups (`702` files, all cross-split) share an **identical label file `SHA-256` hash** across `train`, `valid`, and/or `test`, even when their image `SHA-256` hashes differ. Direct visual inspection of 3 representative cross-split pairs (`6` images) proves that these are **visually indistinguishable re-encoded or minimally augmented copies of the exact same video frame or stock photo** leaked across splits (`VERIFIED`):

| Pair # | Split A Image (`train`) | Split B Image (`test` or `valid`) | Shared Label `SHA-256` Prefix | Image `SHA-256` & Size Comparison | Visual Comparison (`VERIFIED`) | Classification |
|---:|---|---|---|---|---|---|
| **1** | `train/images/video_2022-12-07_23-08-07_mp4-2_jpg.rf.29eb6139e80a248394f6e393ee647f4b.jpg` | `test/images/video_2022-12-07_23-08-07_mp4-2_jpg.rf.1123b4a0d9dcda04bb783bf999e56047.jpg` | `96cd32b7d473...` (`4` files across `train,test`) | `train`: `008e7303b9af0e32...` (`36,281` B)<br>`test`: `033148cd421d49f1...` (`36,284` B) | Both images show the **exact same rotated frame** of a swimmer with raised right arm in a tiled swimming pool (`POOL`), differing only by imperceptible pixel-level JPEG/noise differences. | **`CONFIRMED_SAME_VIDEO`** *(near-exact duplicate frame across `train` & `test`)* |
| **2** | `train/images/video_2022-12-07_23-08-12_mp4-16_jpg.rf.ca89cbac9fdaf234d784fe995327c18d.jpg` | `test/images/video_2022-12-07_23-08-12_mp4-16_jpg.rf.8825911003b4b19eed212eb2803d6d24.jpg` | `6ecc8e06b861...` (`4` files across `train,test`) | `train`: `0cc39bab76c4a1e1...` (`34,589` B)<br>`test`: `8470f71c3f56a4fc...` (`34,589` B) | Both images have the exact same byte size (`34,589` B) and show the **exact same frame** of a person in green-blue pool water with raised left arm (`POOL`). | **`CONFIRMED_SAME_VIDEO`** *(near-exact duplicate frame across `train` & `test`)* |
| **3** | `train/images/val5_png_jpg.rf.866f4e85e8e0d8c47ebfaee67380e767.jpg` | `valid/images/val5_png_jpg.rf.2c939caa81e76a244bbbb3b075774c12.jpg` | `bbf3ac0bbff1...` (`4` files across `train,valid`) | `train`: `234031c7e92bb359...` (`41,069` B)<br>`valid`: `1647ee8f6385a257...` (`41,120` B) | Both images show the **exact same stock photograph** of a silhouetted hand reaching out of open ocean waves against a sunset (`OUT_OF_SCOPE`). | **`CONFIRMED_SAME_VIDEO`** *(near-exact duplicate stock image across `train` & `valid`)* |

### 3.2 Cross-Split Same-Source-Prefix Leakage (`1,616` Cross-Split Source Groups)
Out of `4,949` source groups in `SOURCE_GROUP_INVENTORY.csv`, **`1,616` source groups (`32.7%`) span multiple splits** (`same_source_prefix.csv`). We visually inspected images across splits for **5 multi-split source groups** (`VERIFIED`):

| Source Group Prefix | Splits Present (`train`/`valid`/`test` counts) | Inspected Image Examples Across Splits | Observed Visual Evidence (`VERIFIED`) | Classification |
|---|---|---|---|---|
| `203512036-1-_jpg` | `train` (`3`), `valid` (`1`), `test` (`2`) | `train/.../203512036-1-_jpg.rf.02a7febf04214f45eada62fb78d07fc2.jpg`<br>`valid/.../203512036-1-_jpg.rf.54b30c35e9d63fef7dfeafc73d5ee40f.jpg`<br>`test/.../203512036-1-_jpg.rf.c7be484ea2520c0428627c8befbadc87.jpg`<br>`test/.../203512036-1-_jpg.rf.d8b66eff0d050ad589403464790ec0f1.jpg` (Sheet 4 Tile 3) | All 4 inspected files across `train`, `valid`, and `test` are augmented variants of the **exact same single Bigstock photograph (`Image ID: 203512036`)** of two children in red lifejackets in a swimming pool (`POOL`). `train/02a7febf...` and `test/c7be484e...` are unrotated; `valid/54b30c35...` has the central bounding-box patch horizontally flipped; `test/d8b66eff...` has the central bounding-box patch rotated. | **`CONFIRMED_SAME_VIDEO`** *(same base photograph leaked across all 3 splits)* |
| `drowning47_mp4-10_jpg` | `train` (`12`), `valid` (`2`), `test` (`1`) | `train/.../drowning47_mp4-10_jpg.rf.0376b6a23a2b2140d9bc49eab0a3afea.jpg`<br>`valid/.../drowning47_mp4-10_jpg.rf.732a92457e97fb8d2c033bb47f7a47b5.jpg`<br>`test/.../drowning47_mp4-10_jpg.rf.0d71de9a247bac23ef6f8273f353bed2.jpg` | All 3 files across `train`, `valid`, and `test` are the **exact same video frame** of people in water with rectangular bounding-box patch augmentations on the two foreground heads. | **`CONFIRMED_SAME_VIDEO`** *(same video frame leaked across all 3 splits)* |
| `video40-1-_mp4-81_jpg` | `train` (`11`), `valid` (`1`), `test` (`0`) | `train/.../video40-1-_mp4-81_jpg.rf.08cf8acfc38839eb535f8a464098d0e4.jpg`<br>`valid/.../video40-1-_mp4-81_jpg.rf.e03418891692e0cb40d8f3d05e21865c.jpg` | Both files in `train` and `valid` are the **exact same ocean surf video frame** (`OUT_OF_SCOPE`) of a man swimming in open ocean waves (`train` has a slight bounding-box patch shift on his head; `valid` is unshifted). | **`CONFIRMED_SAME_VIDEO`** *(same video frame leaked across `train` & `valid`)* |
| `drowning_small-1-1000x573_jpg` | `train` (`35`), `valid` (`2`), `test` (`1`) | `train/.../drowning_small-1-1000x573_jpg.rf.021f153a98205013c34d96fb95e7a6ae.jpg` (Sheet 1 Tile 1)<br>`train/.../drowning_small-1-1000x573_jpg.rf.ff5e1fac6bc090eb32782af068bc7868.jpg` (Sheet 1 Tile 3)<br>`test/.../drowning_small-1-1000x573_jpg.rf.c604ab0edcdd9a14a6c37ae3128a87f7.jpg` (Sheet 5 Tile 1) | All files in `train` and `test` are augmented versions (rotated / gray-cutout / black-noise cutout) of the **exact same static photograph** of a child's face splashing in blue water. | **`CONFIRMED_SAME_VIDEO`** *(same base photograph leaked across `train` & `test`)* |
| `-Clipchamp-_mp4-10_jpg` | `train` (`12`), `valid` (`2`), `test` (`1`) | `train/.../-Clipchamp-_mp4-10_jpg.rf.2b6633cd22c8f1c7a890bd0f07da0856.jpg` (Sheet 5 Tile 5)<br>`train/.../-Clipchamp-_mp4-10_jpg.rf.65238cc733fa222480578103974aebde.jpg` (Sheet 5 Tile 6)<br>`train/.../-Clipchamp-_mp4-10_jpg.rf.a355686f8bd23aeb6cc2abec90cdf251.jpg` (Sheet 5 Tile 7) | All 3 inspected `train` tiles (and the group's `valid`/`test` siblings) derive from the **exact same video frame** of two swimmers with pool noodles in a swimming pool with a `"5 FT"` depth marker at top-left (Tile 5 is clean, Tile 6 has black noise cutouts over the swimmers, Tile 7 is blurred). | **`CONFIRMED_SAME_VIDEO`** *(augmented siblings of the same frame)* |

### 3.3 Cross-Prefix Same-Video Leakage (`VERIFIED`)
Even more critically, visual inspection proved that **different source-group prefixes contain frames from the exact same underlying video**:
1. **Indoor Yellow-Green Coping Pool Video Shared Across `youtube-2_jpg`, `youtube-5_jpg`, `youtube-6_jpg`, `youtube-9_jpg`, and `youtube-10_jpg` (`VERIFIED`):**
   - Compare `train/images/youtube-2_jpg.rf.5e4807e549176fb7d19e8f9b13f34313.jpg`, `train/images/youtube-5_jpg.rf.0e193c1be267b71d7e9c16a69b297789.jpg`, `train/images/youtube-6_jpg.rf.81eec4b1de0f1d1ead77a166aae7b901.jpg`, and `train/images/youtube-10_jpg.rf.fde16aa95930dfa2addda0edbd120fda.jpg` (Sheet 2 Tile 12): all four show consecutive frames of the **exact same rescue incident** (a child in a yellow shirt submerged in an indoor swimming pool with yellow-green coping, rescued by a swimmer in a pink cap).
2. **Outdoor Red-Coping Backyard Pool Video Shared Across `youtube-0_jpg`, `youtube-2_jpg`, `youtube-3_jpg`, `youtube-4_jpg`, and `youtube-7_jpg` (`VERIFIED`):**
   - Compare `train/images/youtube-3_jpg.rf.ce3a53732b836d5f0d7e0392f6948e95.jpg` and `train/images/youtube-4_jpg.rf.37bf12bc1d8864de7d4ec6167cae8260.jpg` (plus mosaic quadrants in `youtube-0_jpg`, `youtube-2_jpg`, and `youtube-7_jpg`): all show the **exact same residential pool** with red brick coping and a white lounge chair in the top-right background.
3. **Thai TV News ("THAIRATH TV 32") Backyard Pool CCTV Shared Across `image724_jpg` and `image1944_jpg` (`VERIFIED`):**
   - `test/images/image724_jpg.rf.0c4829c598e629cf37a37e583b44457b.jpg` and `test/images/image1944_jpg.rf.b411cc08a49c0d8d812cc09fc420581b.jpg` are two frames from the **exact same CCTV video broadcast** (showing the same backyard pool, pink float ring, girl in pink on the deck, and adult in blue on the patio).
4. **VisDrone Urban Street Video Shared Across `0000242_00627_d_0000003_jpg` and `0000242_02762_d_0000010_jpg` (`VERIFIED`):**
   - Both prefixes come from VisDrone video sequence `0000242`.
5. **iStock Woman in Blue Water Clip Shared Across `3-727-...-iStock_mp4-2_jpg` and `3-727-...-iStock_mp4-3_jpg` (`VERIFIED`):**
   - All `36` images in `mp4-2` and all `35` images in `mp4-3` are consecutive frames (`mp4-2` and `mp4-3`) from the **exact same iStock video clip**.

> **Implication for Dataset Splitting (`VERIFIED`):** Grouping solely by filename prefix (everything before `_mp4-` or `.rf.`) is **necessary but NOT sufficient** to prevent cross-split leakage, because `image724_jpg` / `image1944_jpg`, `youtube-0_jpg` through `youtube-11_jpg`, and numbered frame prefixes (`10687_jpg`, `11159_jpg`, `11776_jpg`) split consecutive frames of the same video across different source-group names.

---

## 4. Suspected Source Contamination / Non-Homogeneous Prefix Groups (`SUSPECTED_SOURCE`)

### 4.1 Forensic Proof That `youtube-*` Groups Are Composite Mosaic Mixtures (`VERIFIED`)
Task 1 (`Contact Sheets 1–3`, `36` tiles) tested whether source groups such as `youtube-1_jpg` through `youtube-11_jpg` are scene-homogeneous so that a single scene label (`POOL` or `OUT_OF_SCOPE`) could be propagated to all images in the group.

Direct visual inspection of all `36` tiles on Contact Sheets 1–3 (`VERIFIED`) proves that **scene propagation by `youtube-*` source group is IMPOSSIBLE (`SUSPECTED_SOURCE`)**:
- **`31` of the `36` tiles (`86.1%`) on Contact Sheets 1–3 are 2×2 Roboflow Mosaic composites** where a single 640×640 JPEG contains **4 distinct sub-images** stitched together in a 2×2 grid.
- Within a single `youtube-*` source group (and even within a single 640×640 image!), the 4 quadrants mix:
  1. **Indoor/outdoor swimming pools** (green-coping pool, red-coping backyard pool, competition pool lanes).
  2. **Close-up water frames** (`Shutterstock` watermark) where no pool walls or shoreline are visible (`UNCERTAIN`).
  3. **TV news broadcast graphics** (`"MONDAY REPORT: WATER SAFETY"` with a picture-in-picture sign-language interpreter in a dry studio, and `"BENEDICT GALAZAN, HOST / EMR"` standing upright in a blue polo shirt).
  4. **Dry outdoor non-pool scenes** (e.g., child lying on dry grass in `youtube-3_jpg.rf.00732b1f3b4971f02a29ab40219c0aa3.jpg`, Sheet 2 Tile 4; dry outdoor plaza in `youtube-2_jpg.rf.fee60f9060ec8a71932eff3ca823d959.jpg`, Sheet 1 Tile 9).
- Furthermore, within the exact same `youtube-*` group, some images use the **semantic class schema (`4, 5, 6`)** while other images use the **numeric class schema (`0, 1, 2, 3`)** with **synthetic black salt-and-pepper noise cutouts**!

### 4.2 Scene Homogeneity Verdict by Inspected Source Group (`VERIFIED`)

| Source Group | Total Dataset Images (`train`/`valid`/`test`) | Inspected Tiles in Session | Homogeneous Scene? | Scene & Contamination Verdict (`VERIFIED`) |
|---|---:|---:|---|---|
| `drowning_small-1-1000x573_jpg` | `38` (`35`/`2`/`1`) | `4` (Sheet 1 Tiles 1–3, Sheet 5 Tile 1) | Yes (same base image), **No** for quality | All derive from 1 close-up water photo (`UNCERTAIN` scene), but heavily corrupted by `100%` full-frame box, `88.6%` black noise cutout, gray head cutout, and `cls=4` (`Out of Water`) mislabeling. |
| `3-727-...-iStock_mp4-2_jpg` | `36` (`35`/`1`/`0`) | `8` (Sheet 1 Tiles 4–6 + `5` extra malformed) | Yes (same base frame), **No** for quality | All derive from 1 close-up water frame (`UNCERTAIN` scene); contains `6` polygon-only `cls=4` files (`NO_BOX`) and black noise cutout files (`WRONG_OBJECT`). |
| `3-727-...-iStock_mp4-3_jpg` | `35` (`34`/`1`/`0`) | `9` (all `9` malformed files) | Yes (same base frame), **No** for quality | All derive from 1 close-up water frame (`UNCERTAIN` scene); contains `9` mixed box+polygon `cls=4` files including `3` with 90° patch-rotation box misalignment (`SHIFTED`). |
| `youtube-1_jpg` through `youtube-11_jpg` | `2,812` across `11` groups | `42` (`30` on Sheets 1–3 + `2` extra + `10` leakage) | **NO (`SUSPECTED_SOURCE`)** | **Non-homogeneous mixtures** of 2×2 Mosaic composites (`UNCERTAIN`), single-frame pools (`POOL`), close-up water (`UNCERTAIN`), TV studio insets, and black noise cutouts, with `CONFIRMED_EXACT_DUP` and `CONFIRMED_SAME_VIDEO` leakage across `train` and `valid`. |
| `-Clipchamp-_mp4-10_jpg` | `15` (`12`/`2`/`1`) | `3` (Sheet 5 Tiles 5–7) | **Yes (`POOL`)** for scene, **No** for quality | All 3 tiles show the **same swimming pool** with `"5 FT"` marker (`POOL`), correcting prior conflicting `OUT_OF_SCOPE`/`UNCERTAIN` flags; however, Tile 6 (`65238cc7...`) has black noise cutouts over both swimmers (`WRONG_OBJECT`). |
| `0000001_*`, `0000026_*`, `0000103_*`, `0000116_*`, `0000242_*` | VisDrone aerial subset | `6` (Sheet 4 Tiles 4–6 + `3` extra) | **Yes (`OUT_OF_SCOPE`)** | All are VisDrone aerial urban street/traffic scenes (`OUT_OF_SCOPE`) with `cls=1` labeling street pedestrians. |
| `KakaoTalk_20230504_000756264_*` | Korean pool subset | `3` (Sheet 4 Tiles 7–9) | **Yes (`POOL`)** | All are swimming pool survival-floating / swimming instruction scenes (`POOL`), but with partial annotations and conflicting `cls=2` semantics. |

---

## 5. Unresolved Cases (`UNRESOLVED`)

The following items remain `UNRESOLVED` pending a full automated deduplication and visual-clustering pass over all `23,586` images:
1. **Remaining Uninspected Cross-Split Source Groups (`1,605` of `1,616` groups in `same_source_prefix.csv`):**
   - While our visual spot-checks confirmed `CONFIRMED_SAME_VIDEO` leakage in `100%` of the tested multi-split prefixes (`203512036-1-_jpg`, `drowning47_mp4-10_jpg`, `video40-1-_mp4-81_jpg`, `drowning_small-1-1000x573_jpg`, `-Clipchamp-_mp4-10_jpg`, `video_2022-12-07_23-08-07_mp4-2_jpg`, `video_2022-12-07_23-08-12_mp4-16_jpg`, `val5_png_jpg`), the remaining uninspected groups in `same_source_prefix.csv` must be treated as contaminated until re-split by video/cluster.
2. **Sequential Frame Prefixes Without Video Prefix (`10687_jpg`, `11159_jpg`, `11776_jpg`, `130163_jpg`, `130315_jpg`, `image724_jpg`, `image1944_jpg`):**
   - Because each frame number is its own filename prefix, filename-prefix grouping alone cannot link `image724_jpg` and `image1944_jpg` (which we visually proved come from the same Thai TV news CCTV video). Perceptual hashing (`pHash` / `DINOv2` or `CLIP` embeddings) across all `23,586` images is required to group sequential frames from the same video before constructing clean `train`/`valid`/`test` splits.

