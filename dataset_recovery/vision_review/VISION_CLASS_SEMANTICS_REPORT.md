# VISION CLASS SEMANTICS REPORT — `DrowningDetectionTracking_v1`

- **Investigation Date:** 2026-11-03
- **Agent Role:** Visual Inspection & Dataset Forensics Agent (`vision-agent`)
- **Dataset:** `DrowningDetectionTracking_v1` (`23,586` images, `49,040` valid 5-field boxes + `15` segmentation polygon lines across `train`/`valid`/`test`)
- **Project Scope:** Autonomous AI Lifeguard System — **SWIMMING POOLS ONLY**
- **Dataset Approval Status:** **NOT APPROVED FOR TRAINING**

---

## 1. Executive Summary of Class Semantics

`DrowningDetectionTracking_v1` defines **7 classes** in `data.yaml` / `data.fixed.yaml`:
- **Numeric classes (`0`–`3`):** `0 -> '3'`, `1 -> '4'`, `2 -> '5'`, `3 -> '6'` (`24,522` boxes total, **50.00%** of all valid boxes)
- **Semantic classes (`4`–`6`):** `4 -> 'Out of Water'`, `5 -> 'drowning'`, `6 -> 'swimming'` (`24,518` boxes total, **50.00%** of all valid boxes)

### Core Forensic Finding (`VERIFIED`)
Direct pixel-level inspection across **26 independent source groups** proves that **numeric Classes `0`, `1`, `2`, and `3` CANNOT be safely mapped to a single consistent semantic meaning** (`UNKNOWN` final class-semantic status for all four numeric classes):

1. **Multi-Dataset Schema Collision (`VERIFIED`):**
   - **Class `1` (YAML `'4'`)** merges at least two incompatible source datasets:
     - **VisDrone aerial urban traffic dataset** (`0000001_*`, `0000026_*`, `0000103_*`, `0000116_*`, `0000242_*`): Class `1` boxes **urban street pedestrians and motorbike riders** on dry roads and sidewalks (`OUT_OF_SCOPE`).
     - **Pool / CCTV / YouTube mosaic subsets** (`image1944_jpg`, `image724_jpg`, `youtube-0_jpg`, `youtube-10_jpg`): Class `1` boxes **people standing or sitting poolside on the dry deck/edge**.
   - **Class `2` (YAML `'5'`)** merges conflicting behavioral/spatial targets across independent pool groups:
     - In `KakaoTalk_20230504_000756264_04_jpg`, `_10_jpg`, and `_16_jpg` (Contact Sheet 4 Tiles 7–9), Class `2` boxes **children and instructors floating or swimming IN the pool water** (with lifejackets or kickboards).
     - In `image1944_jpg`, `image724_jpg`, and `output_0001_jpg`, Class `2` boxes **adults and children walking, running, or sitting on the DRY pool deck** (`Out of Water`), while people in the water are either labeled Class `0` or left unannotated.
2. **Destructive Pre-Baked Bounding-Box Cutout / Salt-and-Pepper Noise Augmentations (`VERIFIED`):**
   - Across `youtube-*`, `drowning_small-1-1000x573_jpg`, `3-727-...-iStock_mp4-2_jpg`, `65_png_jpg`, and `test1-5-_mp4-12_jpg`, many images labeled with numeric Classes `0`, `1`, `2`, and `3` (and some with Class `6`) have the pixels inside the bounding box **completely obliterated by synthetic black/gray rectangles or black salt-and-pepper noise cutouts** baked directly into the JPEG files.
   - In addition, `60` Class-`0` boxes and `4` Class-`3` boxes are **full-frame (>90% image area) boxes** spanning either rotated whole frames or 90–98% black noise cutouts.
3. **Semantic Classes `4`, `5`, and `6` Are Only `PARTIALLY VERIFIED` Due to Severe Mislabeling & Mosaic Artifacts (`VERIFIED`):**
   - While Classes `4` (`Out of Water`), `5` (`drowning`), and `6` (`swimming`) generally correspond to their YAML names in clean frames, direct visual inspection confirmed severe cross-class contamination:
     - **Class `4` (`Out of Water`)** is applied to a person **completely submerged underwater in a pool** (`youtube-11_jpg.rf.00fe6568dc7256195c70de8daabd0cb5.jpg`, Sheet 3 Tile 7), a child's head **splashing in water** (`drowning_small-1-1000x573_jpg.rf.021f153a98205013c34d96fb95e7a6ae.jpg`, Sheet 1 Tile 1), and all **15 malformed-annotation files** (`3-727-...-iStock_mp4-2` and `mp4-3`) showing a woman **in blue water**.
     - **Class `5` (`drowning`)** is applied to a TV host ("BENEDICT GALAZAN, HOST / EMR") **standing upright on dry land in a blue polo shirt** (`youtube-11_jpg.rf.86eddc8838d48db0a6e0b817aefc7232.jpg`, Sheet 3 Tile 8) and to a **rescue pole / red pool coping edge** (`youtube-2_jpg.rf.043164e22ed59c7332ed7cf722f042f6.jpg`, Sheet 1 Tile 7).
     - **Class `6` (`swimming`)** is repeatedly applied to **picture-in-picture TV sign-language interpreters** inside news broadcast banners (`youtube-1_jpg.rf.8e44ce7a...`, `youtube-6_jpg.rf.8364112c...`, `youtube-7_jpg.rf.f666c656...`, `youtube-8_jpg.rf.1c04fc40...`) and to **black salt-and-pepper noise cutouts** (`-Clipchamp-_mp4-10_jpg.rf.65238cc7...`, Sheet 5 Tile 6).

---

## 2. Summary Table: Class 0–6 Forensic Status

| Class ID | YAML Name | Total Boxes (`train`/`valid`/`test`) | Images Present | Independent Source Groups Visually Inspected | Candidate Meaning(s) Observed in Pixels | Final Semantic Confidence | Safe to Remap / Train As-Is? |
|---:|---|---:|---:|---:|---|---|---|
| `0` | `'3'` | `8,979` (`7,922` / `662` / `395`) | `7,692` | **11** | (a) Person/child head or raised arms in water / drowning (`10_MOV-52`, `129_png`, `124_png`, `134_png`, `203512036-1`, `image1944`, `youtube-10`); (b) Synthetic black/gray noise cutouts (`drowning_small`, `3-727-...-iStock`, `youtube-*`); (c) 60 full-frame (>90% area) boxes (`drowning_small`, `65_png`). | **`UNKNOWN`** | **NO** — Cannot safely merge into `5` (`drowning`) without filtering full-frame boxes, noise cutouts, and non-drowning lifejacket floaters (`203512036-1`). |
| `1` | `'4'` | `2,994` (`2,487` / `284` / `223`) | `1,354` | **8** | **Schema Collision:** (a) Aerial urban street pedestrians & motorbike riders in VisDrone traffic scenes (`0000001_*`, `0000026_*`, `0000103_*`, `0000116_*`, `0000242_*`); (b) Person standing/climbing poolside on dry deck (`image1944`, `image724`, `youtube-0`); (c) Black noise cutouts (`youtube-10`). | **`UNKNOWN`** | **NO** — Contains incompatible VisDrone urban traffic pedestrian labels mixed with poolside deck labels. |
| `2` | `'5'` | `3,389` (`2,850` / `287` / `252`) | `1,304` | **7** | **Schema Collision:** (a) Children/instructors floating or swimming **in pool water** with lifejackets/kickboards (`KakaoTalk_*_04`, `_10`, `_16`); (b) Adults/children walking, running, or sitting **on dry pool deck** (`image1944`, `image724`, `output_0001`); (c) Black noise cutouts (`youtube-4`). | **`UNKNOWN`** | **NO** — Directly contradicts itself across independent pool groups (in-water swimmers/floaters vs dry-deck pedestrians). |
| `3` | `'6'` | `9,160` (`8,044` / `667` / `449`) | `6,603` | **10** | (a) Swimmers/rescuers in open water or pools (`10687`, `11159`, `11776`, `130163`, `130315`, `youtube-10`); (b) Synthetic black/white noise cutouts over swimmers (`youtube-1`, `youtube-3`, `youtube-7`, `test1-5-_mp4-12`). | **`UNKNOWN`** | **NO** — While unoccluded boxes show swimmers, a large fraction of `youtube-*` and `test1-5` Class-`3` boxes point to solid black noise cutouts or open-sea scenes. |
| `4` | `'Out of Water'` | `3,200` (`2,621` / `327` / `252`) | `1,852` | **8** | Intended: person out of water (on pool deck or grass). **Observed Contamination:** also boxes submerged underwater person (`youtube-11`), child head splashing in water (`drowning_small-1`), and woman in blue water across all 15 malformed polygon files (`3-727-...-iStock_mp4-2` & `mp4-3`). | **`PARTIALLY VERIFIED`** | **NO** — Requires purging mislabeled underwater/in-water `Out of Water` boxes and mosaic fragments. |
| `5` | `'drowning'` | `11,226` (`9,828` / `868` / `530`) | `7,909` | **11** | Intended: drowning / distressed / submerged person in water. **Observed Contamination:** also boxes upright TV host on dry land (`youtube-11`), rescue pole/pool coping (`youtube-2`), and truncated mosaic fragments. | **`PARTIALLY VERIFIED`** | **NO** — Requires purging dry-land/object false positives, mosaic fragments, and out-of-scope ocean scenes. |
| `6` | `'swimming'` | `10,092` (`8,815` / `789` / `488`) | `6,227` | **10** | Intended: active swimmer / rescuer in water. **Observed Contamination:** also boxes picture-in-picture TV sign-language interpreters (`youtube-1`, `youtube-6`, `youtube-7`, `youtube-8`), black noise cutouts (`-Clipchamp-_mp4-10`), and truncated mosaic fragments. | **`PARTIALLY VERIFIED`** | **NO** — Requires purging TV broadcast inset boxes, black noise cutouts, and mosaic fragments. |

---

## 3. Detailed Forensic Analysis by Class (Classes 0–6)

### 3.1 Class `0` — YAML Name: `'3'`
- **Dataset Distribution:** `8,979` boxes across `7,692` images (`train`: `7,922`, `valid`: `662`, `test`: `395`). Co-occurs with Class `3` in `3,962` images, Class `1` in `762` images, Class `2` in `652` images, and almost never with semantic Classes `4`–`6` (`0`–`4` images).
- **Independent Source Groups Visually Inspected (`11`):**
  1. `10_MOV-52_jpg` (`test/images/10_MOV-52_jpg.rf.22538ea01ec12d6101cbf6612e56af2e.jpg`, Sheet 4 Tile 1)
  2. `129_png_jpg` (`test/images/129_png_jpg.rf.97afceefe143d1f6197c661ff9d907f6.jpg`, Sheet 4 Tile 2)
  3. `203512036-1-_jpg` (`test/images/203512036-1-_jpg.rf.d8b66eff0d050ad589403464790ec0f1.jpg`, Sheet 4 Tile 3; plus `train` `02a7febf...`, `valid` `54b30c35...`, `test` `c7be484e...`)
  4. `124_png_jpg` (`test/images/124_png_jpg.rf.0bd5d9ff51e7a1be7b78d3cd687dc827.jpg`, extra semantics sample)
  5. `134_png_jpg` (`test/images/134_png_jpg.rf.65bf8da98c6f8fc2fa86ebfdff687028.jpg`, extra semantics sample)
  6. `image1944_jpg` (`test/images/image1944_jpg.rf.b411cc08a49c0d8d812cc09fc420581b.jpg`, extra semantics sample)
  7. `drowning_small-1-1000x573_jpg` (`train` Sheet 1 Tiles 2–3 `9fbbb64c...`, `ff5e1fac...`; `test` Sheet 5 Tile 1 `c604ab0e...`)
  8. `3-727-drowning-stock-videos-and-royalty-free-footage-istock_mp4-2_jpg` (`train` Sheet 1 Tiles 5–6 `532b984c...`, `e2c1502b...`)
  9. `65_png_jpg` (`train` Sheet 5 Tiles 3–4 `1ac055ba...`, `97192378...`)
  10. `youtube-10_jpg` (`train` Sheet 2 Tile 12 `fde16aa9...`; `valid` `04261c56...`)
  11. `youtube-2_jpg` / `youtube-3_jpg` / `youtube-4_jpg` / `youtube-6_jpg` / `youtube-7_jpg` / `youtube-8_jpg` / `youtube-9_jpg` (Sheets 1–3 mosaic tiles)
- **Observed Visual Evidence (`VERIFIED`):**
  - **Pattern A — Person/child in water (distressed, submerged, or raising arms):**
    - In `10_MOV-52_jpg.rf.22538ea0...` (`OUT_OF_SCOPE` murky river/open water), Class `0` tightly boxes a person's head at the surface.
    - In `129_png_jpg.rf.97afceef...` and `134_png_jpg.rf.65bf8da9...` (`OUT_OF_SCOPE` dark open water), Class `0` boxes two raised arms/hands splashing above the water surface.
    - In `124_png_jpg.rf.0bd5d9ff...` (`OUT_OF_SCOPE` dark open water), Class `0` boxes a submerged person in waves.
    - In `image1944_jpg.rf.b411cc08...` (`POOL` Thai TV news "THAIRATH TV 32" backyard pool CCTV), Class `0` boxes the small child submerged/drowning in the pool next to a pink float ring.
    - In `youtube-10_jpg.rf.fde16aa9...` (`POOL` Sheet 2 Tile 12), Class `0` boxes the submerged child in a yellow shirt while Class `3` boxes the swimming rescuer.
  - **Pattern B — Non-drowning child wearing lifejacket in pool (with partial annotation):**
    - In `203512036-1-_jpg` (`POOL`, Bigstock photo 203512036), two children wearing red lifejackets raise their arms in a swimming pool under instructor supervision. Class `0` boxes only the right child (whose rectangular image patch is synthetically rotated in `d8b66eff...` and flipped in `54b30c35...`), leaving the left child completely unannotated.
  - **Pattern C — Obliterated black/gray cutout rectangles & full-frame boxes:**
    - In `drowning_small-1-1000x573_jpg.rf.ff5e1fac...` (Sheet 1 Tile 3), `3-727-...-iStock_mp4-2_jpg.rf.e2c1502b...` (Sheet 1 Tile 6), and numerous `youtube-*` tiles (Sheet 1 Tile 9, Sheet 2 Tiles 3, 6, 8, 9, 11, Sheet 3 Tiles 6, 9, 12), the pixels inside the Class-`0` box are completely replaced by solid gray or black salt-and-pepper noise cutouts.
    - In `drowning_small-1-1000x573_jpg.rf.c604ab0e...` (Sheet 5 Tile 1), Class `0` is a `1.0000` full-frame box (`0 0.5 0.5 1.0 1.0`) covering the entire 640×640 image. In `65_png_jpg.rf.1ac055ba...` and `97192378...` (Sheet 5 Tiles 3–4), Class `0` spans `98.29%` of the frame over a giant black/gray noise cutout (`60` of the `67` oversized >90%-area boxes in the entire dataset belong to Class `0`).
- **Conclusion & Confidence:** **`UNKNOWN`** (Candidate origin: numeric `drowning` / `person_in_water` class from a 4-class numeric schema `[0,1,2,3] -> ['3','4','5','6']` [`INFERENCE`], but heavily corrupted by synthetic black-noise cutouts, 60 full-frame boxes, and non-drowning lifejacket stock photos [`VERIFIED`]).

---

### 3.2 Class `1` — YAML Name: `'4'`
- **Dataset Distribution:** `2,994` boxes across `1,354` images (`train`: `2,487`, `valid`: `284`, `test`: `223`).
- **Independent Source Groups Visually Inspected (`8`):**
  1. `0000001_02999_d_0000005_jpg` (`test` Sheet 4 Tile 4, `28` boxes)
  2. `0000242_00627_d_0000003_jpg` (`test` Sheet 4 Tile 5, `14` boxes)
  3. `0000242_02762_d_0000010_jpg` (`test` Sheet 4 Tile 6, `5` boxes)
  4. `0000026_01000_d_0000002_jpg` (`test` extra semantics sample, `7` boxes)
  5. `0000103_02544_d_0000029_jpg` (`test` extra semantics sample, `1` box)
  6. `0000116_00158_d_0000001_jpg` (`test` extra semantics sample, `7` boxes)
  7. `image1944_jpg` & `image724_jpg` (`test` extra semantics samples, Thai CCTV backyard pool)
  8. `youtube-0_jpg` (`valid` `2c566a4b...`) & `youtube-10_jpg` (`train` Sheet 2 Tiles 9, 11; `valid` `04261c56...`)
- **Observed Visual Evidence (`VERIFIED`):**
  - **Sub-Schema A — VisDrone Aerial Urban Street Pedestrians (`OUT_OF_SCOPE`):**
    - All six inspected `0000*` source groups (`0000001_*`, `0000026_*`, `0000103_*`, `0000116_*`, `0000242_*`) are aerial drone photographs of **urban streets, multi-lane roads, intersections, taxis, storefronts, and sidewalks** from the **VisDrone** benchmark dataset.
    - Every single Class-`1` box in these images encloses a **pedestrian walking on a sidewalk/crosswalk or a person riding a motorbike/bicycle on dry pavement**. There is no water or swimming pool anywhere in these scenes. (In VisDrone's native label format, `1` = `pedestrian`!)
  - **Sub-Schema B — Poolside / Deck Person (`Out of Water` in 4-class numeric pool subset):**
    - In `image1944_jpg.rf.b411cc08...` and `image724_jpg.rf.0c4829c5...` (`POOL`), Class `1` boxes the young girl in pink standing on the **dry pool deck edge** looking into the water.
    - In `youtube-0_jpg.rf.2c566a4b...` (2×2 mosaic), Class `1` boxes a child crouched/climbing on the **red pool coping edge**.
    - In `youtube-10_jpg` (`Sheet 2 Tiles 9 & 11`, and `valid/04261c56...`), Class `1` boxes a synthetic black salt-and-pepper noise cutout positioned over TV host Benedict Galazan standing on the pool deck.
- **Conclusion & Confidence:** **`UNKNOWN`** (`VERIFIED` fatal schema collision between VisDrone `1 = urban pedestrian` and a numeric pool-dataset class for `person on pool edge/deck`).

---

### 3.3 Class `2` — YAML Name: `'5'`
- **Dataset Distribution:** `3,389` boxes across `1,304` images (`train`: `2,850`, `valid`: `287`, `test`: `252`).
- **Independent Source Groups Visually Inspected (`7`):**
  1. `KakaoTalk_20230504_000756264_04_jpg` (`test` Sheet 4 Tile 7, `4` boxes)
  2. `KakaoTalk_20230504_000756264_10_jpg` (`test` Sheet 4 Tile 8, `4` boxes)
  3. `KakaoTalk_20230504_000756264_16_jpg` (`test` Sheet 4 Tile 9, `3` boxes)
  4. `image1944_jpg` (`test` extra semantics sample, `1` Class-`2` box)
  5. `image724_jpg` (`test` extra semantics sample, `1` Class-`2` box)
  6. `output_0001_jpg` (`train/images/output_0001_jpg.rf.3413bb78f9860efbd5219d9ff9ab75aa.jpg`, `2` Class-`2` boxes)
  7. `youtube-4_jpg` (`train` Sheet 2 Tile 6 `c3d6cf9e...`, `2` Class-`2` boxes)
- **Observed Visual Evidence (`VERIFIED`):**
  - **Sub-Schema A — Swimmers / Survival Floaters IN Pool Water (`KakaoTalk_*`):**
    - In `KakaoTalk_20230504_000756264_04_jpg` (`POOL`), `4` of `6` people wearing orange lifejackets floating on their backs in a circle **in the swimming pool water** are boxed as Class `2` (`2` floaters are unannotated).
    - In `KakaoTalk_20230504_000756264_10_jpg` (`POOL`), `4` upright instructors/swimmers **in the indoor pool water** are boxed as Class `2` (while the central floating circle is unannotated).
    - In `KakaoTalk_20230504_000756264_16_jpg` (`POOL`), `3` children wearing swim caps and holding yellow kickboards **in the indoor pool lane** are boxed as Class `2`.
  - **Sub-Schema B — People Walking, Running, or Sitting ON the Dry Pool Deck (`image1944`, `image724`, `output_0001`):**
    - In `image1944_jpg.rf.b411cc08...` and `image724_jpg.rf.0c4829c5...` (`POOL`), Class `2` boxes the adult in a blue shirt **walking/running on the dry tiled patio deck** next to the pool (`Out of Water`), while the child in the water is Class `0` and the girl at the pool edge is Class `1`.
    - In `output_0001_jpg.rf.3413bb78...` (`POOL`), Class `2` boxes **two children on the dry indoor pool deck** (one sitting, one bending over at the edge), while the children swimming in the pool water are unannotated!
  - **Sub-Schema C — Black Noise Cutout in Mosaic (`youtube-4_jpg`):**
    - In `youtube-4_jpg.rf.c3d6cf9e...` (Sheet 2 Tile 6), the Class-`2` boxes cover synthetic black salt-and-pepper noise cutouts at the pool edge.
- **Conclusion & Confidence:** **`UNKNOWN`** (`VERIFIED` direct contradiction across independent pool source groups: Class `2` labels people **in the water** in `KakaoTalk_*` but people **on the dry deck** in `image1944`, `image724`, and `output_0001`).

---

### 3.4 Class `3` — YAML Name: `'6'`
- **Dataset Distribution:** `9,160` boxes across `6,603` images (`train`: `8,044`, `valid`: `667`, `test`: `449`). Co-occurs with Class `0` in `3,962` images.
- **Independent Source Groups Visually Inspected (`10`):**
  1. `10687_jpg` (`test` Sheet 4 Tile 10, `3` boxes)
  2. `11159_jpg` (`test` Sheet 4 Tile 11, `1` box)
  3. `11776_jpg` (`test` extra semantics sample, `2` boxes)
  4. `130163_jpg` (`test` Sheet 4 Tile 12, `1` box)
  5. `130315_jpg` (`test` extra semantics sample, `1` box)
  6. `youtube-10_jpg` (`train` Sheet 2 Tile 12 `fde16aa9...`; Sheet 2 Tiles 9 & 11; `valid` `04261c56...`)
  7. `youtube-1_jpg` (`train` Sheet 1 Tile 12 `ffc95edb...`)
  8. `youtube-3_jpg` (`train` Sheet 2 Tile 3 `247824a0...`)
  9. `youtube-7_jpg` / `youtube-8_jpg` / `youtube-9_jpg` (`train` Sheet 3 Tiles 3, 6, 9, 12)
  10. `test1-5-_mp4-12_jpg` (`test` Sheet 5 Tile 2 `6d95f259...`, `90.4%` area box)
- **Observed Visual Evidence (`VERIFIED`):**
  - **Pattern A — Active Swimmers / Rescuers in Water:**
    - In `10687_jpg`, `11159_jpg`, and `11776_jpg` (`OUT_OF_SCOPE` aerial dark-green open water/sea), Class `3` boxes people swimming in open water.
    - In `130163_jpg` and `130315_jpg` (`UNCERTAIN` letterboxed overhead blue water), Class `3` boxes a person swimming.
    - In `youtube-10_jpg.rf.fde16aa9...` (`POOL` Sheet 2 Tile 12), Class `3` boxes the swimming rescuer in a pink swim cap approaching the submerged child (Class `0`).
  - **Pattern B — Pervasive Black/White Salt-and-Pepper Noise Cutouts & Oversized Boxes:**
    - In `youtube-1_jpg.rf.ffc95edb...` (Sheet 1 Tile 12, competition pool), `youtube-3_jpg.rf.247824a0...` (Sheet 2 Tile 3), `youtube-7_jpg.rf.fea75b71...` (Sheet 3 Tile 3, competition pool with starting block #2), `youtube-8_jpg` (Sheet 3 Tiles 6 & 9), and `youtube-9_jpg` (Sheet 3 Tile 12), **every single Class-`3` box covers a synthetic black or white salt-and-pepper noise cutout** that completely obliterates the underlying swimmer.
    - In `test1-5-_mp4-12_jpg.rf.6d95f259...` (Sheet 5 Tile 2), Class `3` is a giant `90.40%`-area box (`w=0.950, h=0.952`) covering a nearly full-frame black salt-and-pepper noise cutout.
- **Conclusion & Confidence:** **`UNKNOWN`** (Candidate origin: numeric `swimming` class from a prior export [`INFERENCE`], but unapproved for automatic remapping into Class `6` because a massive proportion of Class-`3` boxes in `youtube-*` and `test1-5` enclose synthetic noise cutouts where no swimmer pixels remain [`VERIFIED`]).

---

### 3.5 Class `4` — YAML Name: `'Out of Water'`
- **Dataset Distribution:** `3,200` valid 5-field boxes across `1,852` images (`train`: `2,621`, `valid`: `327`, `test`: `252`), plus `15` segmentation polygon lines in `invalid_annotations.csv`.
- **Independent Source Groups Visually Inspected (`8`):**
  1. `drowning_small-1-1000x573_jpg` (`train` Sheet 1 Tile 1 `021f153a...`)
  2. `3-727-drowning-stock-videos-and-royalty-free-footage-istock_mp4-2_jpg` (`6` malformed-label files in `train`, including Sheet 1 Tile 4 `0123f0e4...`)
  3. `3-727-drowning-stock-videos-and-royalty-free-footage-istock_mp4-3_jpg` (`9` malformed-label files in `train`)
  4. `youtube-11_jpg` (`train` Sheet 3 Tile 7 `00fe6568...`)
  5. `youtube-3_jpg` (`train` Sheet 2 Tiles 1, 2, 4 `000c2294...`, `0011c0c4...`, `00732b1f...`, plus leakage pair `ce3a5373...` / `da35077e...`)
  6. `youtube-4_jpg` (`train` Sheet 2 Tile 5 `20fa9698...`)
  7. `youtube-6_jpg` (`train` Sheet 2 Tile 7 `8364112c...`)
  8. `youtube-7_jpg` (`train` Sheet 3 Tile 5 `f666c656...`)
- **Observed Visual Evidence (`VERIFIED`):**
  - **Intended Use:** In `youtube-3_jpg.rf.ce3a5373...`, `youtube-3_jpg.rf.00732b1f...`, `youtube-4_jpg.rf.20fa9698...`, and `youtube-6_jpg.rf.8364112c...`, Class `4` boxes people on the pool deck, climbing the pool edge, or on dry grass next to a resuscitation scene.
  - **Severe Semantic Contamination (In-Water & Underwater People Labeled `Out of Water`):**
    - In `youtube-11_jpg.rf.00fe6568dc7256195c70de8daabd0cb5.jpg` (Sheet 3 Tile 7), the single Class-`4` (`Out of Water`) box encloses a person **completely submerged underwater** in a swimming pool!
    - In `drowning_small-1-1000x573_jpg.rf.021f153a98205013c34d96fb95e7a6ae.jpg` (Sheet 1 Tile 1), the single Class-`4` (`Out of Water`) box encloses a child's head **splashing in blue water**.
    - In all **15 files** in `3-727-Drowning-Stock-Videos-and-Royalty-Free-Footage-iStock_mp4-2_jpg` and `mp4-3_jpg`, both the 5-field detection boxes (Line 1 of `mp4-3`) and the 22–30 vertex segmentation polygons use Class `4` (`Out of Water`) to label a woman **floating/treading in blue water**.
- **Conclusion & Confidence:** **`PARTIALLY VERIFIED`** (Intended meaning is person out of water / poolside, but contaminated by underwater and in-water mislabeled instances that directly invert the class semantics).

---

### 3.6 Class `5` — YAML Name: `'drowning'`
- **Dataset Distribution:** `11,226` boxes across `7,909` images (`train`: `9,828`, `valid`: `868`, `test`: `530`).
- **Independent Source Groups Visually Inspected (`11`):**
  `youtube-1_jpg`, `youtube-2_jpg`, `youtube-3_jpg`, `youtube-4_jpg`, `youtube-5_jpg`, `youtube-6_jpg`, `youtube-7_jpg`, `youtube-9_jpg`, `youtube-10_jpg`, `youtube-11_jpg`, `drowning47_mp4-10_jpg`, `video40-1-_mp4-81_jpg`.
- **Observed Visual Evidence (`VERIFIED`):**
  - **Intended Use:** Boxes submerged, struggling, or passive people in water (e.g., child submerged in green-edged indoor pool in `youtube-2`, `youtube-5`, `youtube-6`; woman head in water in `youtube-1_jpg.rf.022527c1...`; person splashing in red-coping pool in `youtube-4_jpg.rf.37bf12bc...`).
  - **Observed Contamination:**
    - In `youtube-11_jpg.rf.86eddc8838d48db0a6e0b817aefc7232.jpg` (Sheet 3 Tile 8, bottom-left quadrant), Class `5` (`drowning`) boxes TV host **Benedict Galazan standing upright on dry land in a blue polo shirt** (`WRONG_OBJECT`).
    - In `youtube-2_jpg.rf.043164e22ed59c7332ed7cf722f042f6.jpg` (Sheet 1 Tile 7), Class `5` boxes a **rescue pole and red pool coping edge** rather than a person (`WRONG_OBJECT`).
    - In `video40-1-_mp4-81_jpg` (`train` and `valid`), Class `5` boxes a man in open ocean surf (`OUT_OF_SCOPE`).
- **Conclusion & Confidence:** **`PARTIALLY VERIFIED`** (Represents drowning/distressed person in water in clean frames, but requires filtering dry-land false positives, non-person boxes, mosaic seam truncations, and open-ocean scenes).

---

### 3.7 Class `6` — YAML Name: `'swimming'`
- **Dataset Distribution:** `10,092` boxes across `6,227` images (`train`: `8,815`, `valid`: `789`, `test`: `488`).
- **Independent Source Groups Visually Inspected (`10`):**
  `-Clipchamp-_mp4-10_jpg` (Sheet 5 Tiles 5–7), `youtube-1_jpg`, `youtube-2_jpg`, `youtube-3_jpg`, `youtube-4_jpg`, `youtube-5_jpg`, `youtube-6_jpg`, `youtube-7_jpg`, `youtube-9_jpg`, `youtube-11_jpg`, `drowning47_mp4-10_jpg`.
- **Observed Visual Evidence (`VERIFIED`):**
  - **Intended Use:** Boxes active swimmers or rescuers in water (e.g., two swimmers holding pool noodles in `-Clipchamp-_mp4-10_jpg.rf.2b6633cd...` and `a355686f...` on Sheet 5 Tiles 5 & 7; lap swimmers and rescuers in `youtube-*`).
  - **Observed Contamination:**
    - **TV Sign-Language Interpreter Boxes (`WRONG_OBJECT`):** In `youtube-1_jpg.rf.8e44ce7a...` (Sheet 1 Tile 11), `youtube-6_jpg.rf.8364112c...` (Sheet 2 Tile 7), `youtube-7_jpg.rf.f666c656...` (Sheet 3 Tile 2), and `youtube-7_jpg.rf.f666c656...` / `youtube-7` (Sheet 3 Tile 5), Class `6` (`swimming`) boxes the **sign-language interpreter standing in a dry studio inset** inside the "MONDAY REPORT: WATER SAFETY" TV broadcast graphic!
    - **Black Noise Cutouts (`WRONG_OBJECT`):** In `-Clipchamp-_mp4-10_jpg.rf.65238cc733fa222480578103974aebde.jpg` (Sheet 5 Tile 6), both Class-`6` boxes cover **synthetic black salt-and-pepper noise cutouts** that completely obliterate the two swimmers.
- **Conclusion & Confidence:** **`PARTIALLY VERIFIED`** (Represents active swimmers in clean frames, but contaminated by TV broadcast sign-language inset boxes, synthetic black noise cutouts, and 2×2 mosaic seam truncations).

---

## 4. Why Previous AI Hypotheses Must Not Be Used for Blind Remapping

In `GEMINI_REVIEW_MAPPING_REPORT.md` (`PREVIOUS_AI` / `UNVERIFIED`), a prior text note suggested that numeric classes `0`–`3` might map to a 4-class schema or to `Out of Water`/`drowning`/`swimming`. Our direct pixel inspection (`VERIFIED`) refutes any blind numeric-to-semantic remapping:
1. Mapping Class `1` (`'4'`) to `Out of Water` would inject **thousands of VisDrone aerial urban traffic pedestrian boxes** (`0000*`) into the lifeguard dataset.
2. Mapping Class `2` (`'5'`) to either `Out of Water` or `swimming`/`drowning` would corrupt training labels because Class `2` is used for **in-water swimmers/floaters** in `KakaoTalk_*` AND for **dry-deck pedestrians** in `image1944_jpg`, `image724_jpg`, and `output_0001_jpg`.
3. Mapping Class `0` (`'3'`) or Class `3` (`'6'`) into `drowning` or `swimming` without per-image visual filtering would inject **hundreds of solid black/gray salt-and-pepper noise cutout boxes** and **64 full-frame (>90% area) boxes** directly into model training.

> **Final Determination:** Classes `0`, `1`, `2`, and `3` remain **`UNKNOWN`** at the dataset level and **MUST NOT** be remapped by script. The dataset remains **NOT APPROVED FOR TRAINING**.

