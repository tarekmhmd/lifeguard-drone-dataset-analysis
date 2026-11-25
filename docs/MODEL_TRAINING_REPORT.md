# Model training report — pool detectors (Z5cc and ddt) and system integration

Status date: 2026-11-25. This document records **everything done to the data and the models**, in the order
it was done, including the experiments that failed and the reasons for every decision. Numbers are copied
from the training notebooks and evaluation runs; nothing here is estimated unless marked "approx.".

Related documents: [`DEMO.md`](DEMO.md) (how to run the demo), [`TEMPORAL_PLAN.md`](TEMPORAL_PLAN.md)
(next model), [`ROADMAP.md`](ROADMAP.md) (project status), [`SYSTEM_GUIDE.md`](SYSTEM_GUIDE.md).

---

## 0. Summary

| Model | Camera | Data | Final test result (held-out videos) | Weights | Config |
| --- | --- | --- | --- | --- | --- |
| **Z5cc** YOLO11s, imgsz 448 | side-view pool camera | Z5cc_v2 (49 train+val videos) | mAP50 **0.704**; drowning AP50 0.649, P 0.737, **R 0.633** | `models/z5cc_s448_final.pt` | `configs/ai_z5cc.yaml` |
| **ddt** YOLO11s, imgsz 640 | overhead / high camera | ddt_v2 (v1 + 482 human-reviewed images) | mAP50 **0.636**; swimming AP50 0.845; drowning AP50 0.427, P 0.581, R 0.462 | `models/ddt_v2_final.pt` | `configs/ai_ddt.yaml` |

System level (Z5cc + new detector-distress PRE_ALERT rule, 12 test clips): **5 / 6 near-drowning clips
raised a PRE_ALERT** after 1.9–8.0 s and **0 / 6 normal-swimming clips** did. Twelve clips are a small sample.

Main lessons:
1. **More videos beat a bigger model.** Adding the 12 validation videos to Z5cc training raised test mAP50
   by 7.7 points in one step; a larger model or stronger augmentation gave less or nothing.
2. **What helps depends on the data.** A larger input (448 vs 224) helped Z5cc (small people, clean crops)
   and hurt ddt (960 vs 640; low-quality upscaled frames).
3. **Frame-level drowning accuracy is limited by definition.** Many "drowning" frames look like swimming in
   a single image. Event-level (over 2 s) the same detector catches most episodes, which is why a temporal
   layer is the next step.
4. **The `drowning` label stays human-only.** Every drowning box added to ddt was decided by a named human
   reviewer, one decision per source video, with model suggestions only as an aid.

---

## 1. Z5cc (side-view pool video)

### 1.1 Source data

The available copy is the "smart subset" of Z5cc_v1 (`MyDrive/Colab Notebooks/Z5cc_v1`): frames cropped to
**224 × 224** from CVAT-annotated YouTube pool clips, 30 fps, classes `[swimming, drowning]`, already split
**by video**. The raw videos and CVAT XML were not available, so re-extracting at a higher resolution was
not possible. No earlier weights existed, so every number below comes from models trained in this work.

### 1.2 Audit (read-only)

| | train | val | test |
| --- | --- | --- | --- |
| images | 29,109 | 6,830 | 6,631 |
| images without a label file | 0 | **165** | 0 |
| empty label files (true background) | 205 | 145 | 78 |
| boxes swimming / drowning | 44,804 / 13,496 | 10,340 / 5,185 | 10,602 / 2,338 |
| drowning share | 23.1 % | 33.4 % | 18.1 % |
| videos / clips | 37 / 57 | 12 / 14 | 6 / 12 |
| consecutive near-duplicates (dHash ≤ 3) | 26 % | 36 % | 32 % |
| box width at 224 px (p10 / median / p90) | 22.9 / 54.0 / 102.5 | 18.1 / 40.3 / 83.7 | 19.7 / 45.9 / 117.3 |

- **Video leakage between splits: none.** Label flips (same person, IoU ≥ 0.5, class changes between
  consecutive frames): **0**. All images 224 × 224.
- The 165 unlabelled val frames are the tail of one clip (`zuZIfy4aBEY`). Visual check: every frame shows a
  child in the water, so they are **missing annotations, not background**. Kept as background, they would
  penalise correct detections.
- Review sheets (`Z5cc_work/audit_20261124_203720`) showed mostly correct boxes, with two caveats:
  the box convention differs between videos (head only vs whole body), and many `drowning` boxes mark a
  person almost invisible under water — correct from the video, very hard from one frame.

### 1.3 Z5cc_v2 (cleaning)

| Rule | Effect |
| --- | --- |
| train: drop a frame if dHash ≤ 3 to the last kept frame of the same clip **and** its boxes are unchanged (±0.05) **and** fewer than 5 frames were dropped in a row | 29,109 → **22,884** (swim 33,468 / drown 10,926; drowning share 24.6 %) |
| val: drop images without a label file | 6,830 → **6,665** |
| test: unchanged (verified by the build) | **6,631** |

The build ran as a Colab cell in `E:\lifeguardDron\notebooks\driveConnect.ipynb` (logic tested locally on
synthetic data first). Output: `Z5cc_work/z5cc_v2.zip` (261 MB) with `MANIFEST.json` listing every dropped
image and its reason.

### 1.4 Experiments (Colab, Tesla T4, ultralytics 8.4.218, seed 42)

Evaluation: the test split (6 videos never used for training), `best.pt` unless stated. Drowning P/R are
Ultralytics' values at the max-F1 confidence.

| Run | Model / imgsz | Settings | Epochs (time) | Best epoch | val mAP50 | **test mAP50** | mAP50-95 | swim AP50 | **drown AP50** | drown P | **drown R** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| n224 (baseline) | YOLO11n / 224 | defaults, batch 64 | 43 (0.69 h) | 18 | 0.623 | 0.571 | 0.286 | 0.699 | 0.443 | 0.509 | 0.509 |
| s448 | YOLO11s / 448 | mixup 0.1, patience 15 | 36 (1.96 h) | 21 | 0.607 | **0.627** | 0.318 | 0.738 | 0.516 | 0.549 | 0.579 |
| s448r | YOLO11s / 448 | + freeze 10, mixup 0.2, scale 0.7, translate 0.2, hsv_s 0.8, hsv_v 0.5 | 17 (0.68 h) | 2 | 0.548 | 0.607 | 0.292 | 0.685 | 0.529 | 0.590 | 0.511 |
| **final** | YOLO11s / 448 | s448 settings on **train + val**, 25 fixed epochs, `last.pt` | 25 (1.73 h) | 25 | (trained on) | **0.704** | **0.330** | **0.758** | **0.649** | **0.737** | **0.633** |

Loss, first → last epoch (from each run's `results.csv`):

| Run | train box | train cls | val box | val cls |
| --- | --- | --- | --- | --- |
| n224 | 1.67 → 1.18 | 1.64 → 0.66 | 1.89 → 1.81 | 1.82 → 1.47 |
| s448 | 1.55 → 0.93 | 1.66 → 0.55 | **1.67 → 1.94** (rises: overfitting) | 2.41 → 1.75 |
| s448r | 1.59 → 1.27 | 1.66 → 0.89 | 1.72 → 1.68 | 3.01 → 1.77 |
| final | 1.51 → 0.74 | 1.58 → 0.34 | 1.48 → 0.67 (val was trained on) | 2.84 → 0.30 |

What each run taught us:
- **n224.** Val reached 0.623 at epoch 18 and then stayed around 0.59 while the training loss kept falling:
  the model memorises the 37 training videos.
- **s448.** A larger input helped on test (+5.6 mAP50, +7 points drowning recall) even though val did not move:
  the people are small (median width 54 px at 224).
- **s448r.** Freezing the backbone stopped the model from adapting to people in water (best epoch 2).
  Rejected. Full fine-tuning is required on this data.
- **final.** Same settings as s448 with the 12 validation videos added (49 videos): the largest single gain.
  Its val number is not meaningful (val was trained on), so the epoch count was fixed from s448.
- Confusion matrix of the final model (conf 0.25): drowning → drowning 0.71, → swimming 0.16, missed 0.13;
  swimming found 0.90. The main error is **drowning classified as swimming**, i.e. a single-frame limit.

Honest note: s448 was chosen over s448r partly from test numbers. The differences were large, but the test
split is therefore not perfectly untouched; the report states "evaluated on 6 videos never used for training".

### 1.5 Augmentation used (online, every run)

Mosaic 1.0, horizontal flip 0.5, HSV jitter, scale, translate, random erasing 0.4, RandAugment, light blur and
CLAHE; mixup 0.1 for s448 and final. **Vertical flip and rotation off**: a fixed side-view camera never sees
the pool upside down. Offline copies were not used (online augmentation gives a new variant every epoch).

---

## 2. ddt (overhead / top-view images)

### 2.1 ddt_v1 (before this work)

`DrowningDetectionTracking_v2_clean/dataset/ddt_v1`: 925 images (652 / 137 / 136), 640 × 640, classes
`[swimming, drowning, out_of_water]`, produced by the audited cleaning pipeline of that repository.
**It contains 0 drowning boxes**: every image with a possible drowning was moved to
`quarantine/ddt_v1/DROWNING_CANDIDATE` (1,466 images) because `drowning` may only be set by a named human.

Local training (GTX 1650, YOLO11n, imgsz 640, 100 epochs): test swimming mAP50 0.738, mAP50-95 0.296,
P 0.78, R 0.644; val box loss stayed flat around 2.3 (suspected inconsistent boxes).

### 2.2 Box-convention audit (`scripts/box_unify_audit.py`, read-only)

COCO YOLO11x compared with every labelled box (1,880):

| category | swimming | share |
| --- | --- | --- |
| MATCH (IoU ≥ 0.5) | 450 | 24 % |
| PARTIAL (label = head / upper body only) | 60 | 3 % |
| OVERSIZE (label looser than the person) | 224 | 12 % |
| SHIFTED | 74 | 4 % |
| NO_DET (detector found nobody) | 1,052 | 57 % |
| unlabelled confident people | 51 | |

The review sheets show that **NO_DET is the COCO detector failing on top-view swimmers**; those labels are
correct. Automatic re-boxing with a COCO model would have destroyed more than half of the good labels, so
the planned "box unification" step was **dropped**. The head-only problem is small (3 %).

### 2.3 Drowning-candidate triage (`scripts/drowning_candidate_triage.py`, read-only)

1. **188** images whose labelled person is hidden by a Roboflow noise cut-out were excluded (image quality,
   not a class decision).
2. The remaining **1,278** images were grouped by source clip (`SourceNamer`): **157 groups**.
3. The Z5cc final model wrote a per-box suggestion (drowning / swimming / "?" when no model box overlaps).
4. One contact sheet per group (`review/ddt_v1/drowning_triage/sheets/G###.jpg`) plus `REVIEW.html`.

The **24 groups with ≥ 5 images (1,071 images, 84 %)** were reviewed by **tarekmhmd** (one decision per
group; 133 small groups were not reviewed and are not used):

| Decision | Meaning | Groups |
| --- | --- | --- |
| DROWN | single-person images → drowning; multi-person images skipped | 18 (G001–G012, G017, G018, G020–G023) |
| SWIM | every box → swimming | 4 (G013, G014, G015, G019) |
| ACCEPT | per-box model suggestions confirmed; images with "?" skipped | 2 (G016, G024) |

Decisions and the reviewer's notes are in `review/ddt_v1/drowning_triage/GROUP_DECISIONS.csv`.

### 2.4 ddt_v2 build (`scripts/build_ddt_v2.py`; ddt_v1 and the quarantine are not modified)

| Filter | Images removed |
| --- | --- |
| group not reviewed or EXCLUDE | 207 |
| DROWN group, image with several people | 107 |
| dark noise cut-out over a box (dark fraction ≥ 0.35) | 138 |
| grey noise cut-out over a box (≥ 60 % colourless pixels) | 92 |
| cap of 60 evenly spaced frames per group | 251 |
| ACCEPT group, unmatched box | 1 |

The grey filter was added after the first build: a random sample of 24 added images still showed 3 grey
noise squares labelled `drowning`. After the fix a sample of 36 added images was clean.

Groups already present in ddt_v1 keep their split; new groups go whole to the split with the largest deficit
(70 / 15 / 15), so **no clip is in two splits**. Every added image is logged in
`dataset/ddt_v2/ADDED_FROM_TRIAGE.csv` (group, decision, reviewer, split, classes).

| split | images | swimming | **drowning** | out_of_water |
| --- | --- | --- | --- | --- |
| train | 971 | 1,315 | **299** | 19 |
| valid | 219 | 297 | **66** | 1 |
| test | 217 | 313 | **52** | 0 |

### 2.5 Experiments (Colab, Tesla T4; notebook `notebooks/ddt_v2_train.ipynb`)

Augmentation: as Z5cc but **vertical flip 0.5 and rotation ±10° on**, because an overhead view has no fixed
"up". Mixup 0.1.

| Run | imgsz | Settings | Epochs (time) | Best epoch | **test mAP50** | mAP50-95 | swim AP50 | **drown AP50** | drown P | **drown R** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| s640 | 640 | 3 classes, patience 30 | 110 (0.45 h) | 80 | 0.582 | 0.213 | 0.695 | **0.470** | **0.717** | 0.438 |
| s960 | 960 | classes [0,1] only, 150 epochs, patience 50 | 150 (1.32 h) | 134 | 0.427 | 0.163 | 0.683 | 0.171 | 0.243 | 0.321 |
| **final** | 640 | s640 settings on **train + valid**, 80 fixed epochs, `last.pt` | 80 | 80 | **0.636** | **0.250** | **0.845** | 0.427 | 0.581 | **0.462** |

Loss, first → last epoch: s640 train box 2.14 → 1.32, cls 2.97 → 0.87, val box 2.48 → 2.29, cls 6.11 → 2.13;
s960 train box 2.15 → 1.21, cls 3.33 → 0.60, val box 2.49 → 2.29, cls 4.80 → 2.37; final train box
2.10 → 1.35, cls 2.78 → 0.69 (val 2.45 → 1.61 / 3.59 → 0.78, val was trained on).

- **s640**: the first ddt model that detects drowning at all. Its val mean (0.33) was pulled down by
  `out_of_water` (1 val instance, AP 0), which also made best-epoch selection noisy.
- **s960** was clearly worse on drowning on **both** val (0.289 → 0.189) and test. Likely cause: the
  drowning frames are low-quality video frames; upscaling them further enlarges artefacts. Several settings
  changed at once (size, classes, epochs, batch 8), so the exact cause is not isolated. Rejected.
- **final** is the deliverable: much better swimming (+15 AP50 on 313 boxes); drowning AP and recall
  moved by a few points in opposite directions on only 52 boxes, which is within noise.

---

## 3. System integration (code in this repository)

| Change | Where | Why |
| --- | --- | --- |
| `yolo.person_class_ids` (several model classes → PERSON) | `ai/config.py`, `ai/perception/factory.py` | pool models have `swimming` and `drowning`; the detector only localises people |
| class-agnostic NMS when several classes map to PERSON | `ai/detection/ultralytics_detector.py` | per-class NMS can give one person a swimming **and** a drowning box → two tracks (seen: 3 → 2 tracks on a test clip) |
| `Detection.distress` → `TrackedBox.distress` → feature `det_distress_2s` | `ai/detection/base.py`, `ai/tracking/bytetrack.py`, `ai/perception/real_engine.py`, `contracts/types.DISTRESS_FEATURE` | carries the detector's drowning confidence per person, averaged over 2 s |
| risk rule `risk.detector_distress_threshold` → PRE_ALERT, reason `DETECTOR_DISTRESS`; never ALARM by itself; PRE_ALERT held while it persists | `system/risk/config.py`, `fsm.py`, `engine.py`, `contracts/enums.RiskReason` | early warning while the swimmer is still visible (before any submersion) |
| configs | `configs/ai_z5cc.yaml`, `configs/ai_ddt.yaml`, `configs/system_demo_drone.yaml` | threshold 0.5 only in the demo config; `system.yaml` keeps it `null` (off) |
| dashboard: invisible characters stripped from pasted paths; config defaults from `LIFEGUARD_AI_CONFIG` / `LIFEGUARD_SYSTEM_CONFIG` | `app/dashboard_controller.clean_input`, `app/dashboard.py` | paths copied from an Arabic PDF carried bidi marks ("video not found") |

### 3.1 How the early-warning rule was settled (disclosed)

Measured on the 12 Z5cc **test** clips (6 near-drowning, 6 normal swimming), threshold **0.5 fixed in
advance** and never changed:

| Version of the 2 s mean | near-drowning caught | false PRE_ALERT on swimming |
| --- | --- | --- |
| mean from the track's first frame (window not yet full) | 6 / 6 | 3 / 6 |
| full 2 s window required (track ≥ 1.9 s old), matched frames only | 6 / 6 | 1 / 6 |
| **full window, unmatched (lost) frames count as 0** (final) | **5 / 6** | **0 / 6** |

The window rules were adjusted after looking at these clips, so this table is **evidence, not a
validation**; it must be re-measured on new recordings. Offline, the 2 s mean separated the clips
completely (drowning clips reached 0.51–0.86, swimming clips at most 0.21).

### 3.2 End-to-end checks

- Z5cc: 300-frame test clip → 0 errors, about **14 ms per frame** on a GTX 1650 (real time).
- Dashboard (browser): ALARM by submersion on `demo_06`, early PRE_ALERT about 2 s into `demo_11`,
  Confirm → ALARM → Confirm → DISPATCH → `[SIMULATION] mission-0001 started`.
- ddt: drowning test clip → PRE_ALERT (`DETECTOR_DISTRESS`); swimming test clip → none. Both clips are built
  from separate frames of one source clip, so this checks the pipeline, not alarm accuracy.
- Known artefact: on 224 px crops, swimmers leaving the frame look like submersions (`demo_01`, `demo_07`
  raise a submersion ALARM). A calibrated full-pool camera removes this.
- Tests: the full suite passes with these changes (1 GUI test skipped); ruff clean.

---

## 3.3 Temporal behaviour model (first training, not adopted)

A softmax behaviour model was trained on the tracked features of the 14 Z5cc val clips. Those features were
produced with the s448 detector, which did not see these clips. The model was tested on the 12 test clips.
Results:

- **Sample level:** precision 0.57–0.75 and recall 0.74–0.83, against the detector rule's precision 1.00 and
  recall 0.62.
- **Episode level:** it caught 6/6 episodes but raised false alerts on 12–30 of 35 swimmer tracks. The
  deployed detector rule caught 5/6 with 0/35.
- **Motion-only features:** close to chance (precision 0.25).

It is therefore **not enabled**. Full table and reasons: [`TEMPORAL_PLAN.md`](TEMPORAL_PLAN.md) section 8.
The reasons are uncalibrated 224 px crops, static swimmers on floats, and only 14 training clips.

## 4. Engineering notes (Colab and data handling)

- Training ran on Colab (T4) from Cursor through the Colab extension. The Drive mount works only with the
  **same Google account as the extension**; the data folder `Z5cc_work` was shared from the data account and
  added as a shortcut in the subscription account's My Drive.
- Copying ~85,000 small files from Drive took about 1 h and sometimes failed with `OSError [Errno 5]` while
  listing large folders. Fix: a parallel, resumable copy with retries once, then **one zip per dataset**
  (`z5cc_v1.zip`, `z5cc_v2.zip`, `ddt_v2.zip`) — restoring takes under a minute.
- Runs were written straight to Drive (`Z5cc_work/runs`, `runs_ddt`) so a disconnect resumes from `last.pt`;
  results were appended to `RESULTS_test.csv` / `RESULTS_ddt_test.csv`.
- Ultralytics in this version picks `best.pt` by mAP50-95; the first result table reported the mAP50-max
  epoch instead (s448 showed epoch 31 / 0.615, the real `best.pt` is epoch 21 / 0.607). Fixed in cell 6.

## 5. Where everything is

| What | Path |
| --- | --- |
| Z5cc training notebook (with outputs) | `E:\lifeguardDron\notebooks\z5cc_v2_train.ipynb` |
| ddt training notebook (with outputs) | `E:\lifeguardDron\notebooks\ddt_v2_train.ipynb` |
| Drive backup (data zips, all runs, results tables, audit sheets) | `E:\lifeguardDron\Z5cc_work_backup\Z5cc_work\` |
| ddt scripts | `DrowningDetectionTracking_v2_clean/scripts/box_unify_audit.py`, `drowning_candidate_triage.py`, `build_ddt_v2.py` |
| ddt review material | `DrowningDetectionTracking_v2_clean/review/ddt_v1/box_unify/`, `review/ddt_v1/drowning_triage/` |
| ddt_v2 dataset | `DrowningDetectionTracking_v2_clean/dataset/ddt_v2/` (+ `E:\lifeguardDron\colab_upload\ddt_v2.zip`) |
| final weights (git-ignored) | `models/z5cc_s448_final.pt`, `models/ddt_v2_final.pt` |
| demo videos and guides | `E:\lifeguardDron\demo\` (`DEMO_PATHS.pdf/.docx`, `PATHS_COPY.txt`, `run_demo.sh`) |

## 6. Limitations and next steps

- Evaluation is on internet clips (Z5cc: 224 px crops; ddt: mixed sources), not on the team's own pool
  camera. False alarms per hour cannot be measured on these clips.
- The ddt test split has only 52 drowning boxes from a few clips; differences of a few points are noise.
- ddt: 133 small candidate groups and the multi-person drowning images were not used; adding them needs
  per-box human review.
- The temporal model was trained once and not adopted (section 3.3). Next: calibrate a real camera and record
  staged sessions at a real pool (same camera and angle as deployment), then re-run
  the temporal training (development script `z5cc_temporal_experiment.py`) on that data.
