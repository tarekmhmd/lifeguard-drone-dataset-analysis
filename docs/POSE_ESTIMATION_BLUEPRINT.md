# Pose Estimation Blueprint — Autonomous AI Lifeguard System

Version 1.0 · 2026-11-27 · Scope: all human pose estimation (HPE) work on the pool lifeguard assistant to date.

**Legend.** **[M]** measured in this project (file or command given) · **[H]** historical result from an earlier
run, reported in an existing document or notebook output · **[A]** assumption or estimate · **[P]** plan or
recommendation (not done).

> **Status in one paragraph.** The official, unmodified, COCO-pretrained **Ultralytics YOLO11s-pose** is integrated
> into the perception pipeline and draws skeletons on tracked swimmers in the dashboard. It is **display and recording
> only**: the risk engine does not read keypoints, there is no pose weight, and pose cannot raise WATCH, PRE_ALERT or
> ALARM. Experimental fine-tuning (distillation from a SwimXYZ-tuned ViTPose teacher and three pseudo-label variants)
> did **not** show a clear improvement over the pretrained model against an independent reference, so the pretrained
> model was kept. Real-pool pose accuracy has **not** been established: no sufficiently independent human-annotated
> validation set exists yet. Pose does not, and cannot by itself, establish drowning or distress.

---

## 1. Project objective and motivation

The project is a swimming-pool lifeguard **assistant**: video → person detection and tracking → temporal risk
assessment → incidents and alerts for lifeguards → optional, simulation-only drone response. It raises suspected
incidents for a human to check; it never declares a drowning and never acts physically on its own (README, master
Engineering Blueprint §1).

The pose work had one objective: give each tracked swimmer a body-keypoint description (head, shoulders, arms, hips,
legs) that later, once validated, could feed posture features into the temporal analysis — without weakening any
existing safety behaviour.

## 2. Why human pose estimation was added

The master Engineering Blueprint (2026-11-23, "Engineering Blueprint5.pdf") states the reasoning [H]:

- §1: the drowning decision "must come from a sequence of per-track features (head height, vertical motion,
  posture, time submerged), never from one frame or one box class".
- §2.1: the temporal classifier should be fed features including "posture angle from pose keypoints".
- §M4 "Pose estimation (optional, recommended)": purpose "posture and arm-motion features"; start pretrained
  (YOLO11-pose); "expect head, shoulders and arms to be usable and the legs to be mostly invisible"; fine-tune only if
  pretrained keypoints fail on swimmers; "pose adds value only if the upper body is reliably visible".
- §4: only three components may need training, the pose model "optionally (fine-tune only if pretrained keypoints
  fail on swimmers)"; the open question "Do pose features add value over box-and-motion features?" remains open.

Pose was therefore added as **evidence for later features**, not as a detector of drowning. Instinctive drowning
behaviour (vertical body, head back, arms pressing down) is posture-like, which is why pose is a plausible future
input; whether it helps is untested (§20).

## 3. Relationship between detection, tracking and pose

| Stage | Answers | Component | Changed by the pose work? |
| --- | --- | --- | --- |
| Person detection | Where is a person in this frame? | YOLO11s pool detector (Z5cc side-view `z5cc_s448_final.pt`, or ddt overhead) | No |
| Tracking | Which person is which over time? | ByteTrack (`bytetrack_np_v1`) | No |
| Pose estimation | Where are this person's joints? | YOLO11s-pose on the **tracked box crop** | New (display only) |
| Temporal features / submersion | Is this swimmer static, invisible, submerged too long? | Feature extractor, submersion monitor | No |
| Risk | Should a lifeguard look? | Rule-based risk engine + FSM | No |

Pose depends on tracking (it only runs on boxes ByteTrack reports as visible) and never feeds back into detection,
tracking or risk.

## 4. Implemented architecture [M]

```
video frame
   │
   ├─► person detector (Z5cc YOLO11s) ──► detections ──► ByteTrack ──► tracked boxes (ids)
   │                                                                     │
   │          visible (TRACKED) boxes, highest detector confidence first │
   │          ┌──────────────────────────────────────────────────────────┘
   │          ▼
   │   crop = box + 15 % margin (clipped)  ──►  ONE batched YOLO11s-pose predict over all crops
   │          │
   │          ▼
   │   per crop: pose person box with best IoU to the tracked box (≥ 0.3), else none
   │          │   keypoints: crop pixels + crop origin → frame pixels; undetected / out-of-frame → (0,0,0)
   │          ▼
   └────► TrackObs(track_id, box, features, keypoints=17×(x,y,conf)) ──► risk engine (ignores keypoints)
                                                     │
                                                     └──► dashboard: boxes (risk colour) + skeleton (part colour)
```

Design choices and why:

- **Crops, not a full-frame second detector.** The master blueprint (§M4) suggested full-frame pose matched by IoU;
  the integration instead runs pose on crops of the tracked boxes, as required by the integration brief ("do not run
  an unnecessary second person detector"). The pose model's own person boxes are used only to pick the right person
  inside a crop.
- **Never invent joints.** Undetected, non-finite or out-of-frame points are `(0, 0, 0)`; nothing is moved.
- **Failure isolation.** Any pose exception or malformed output leaves `keypoints = None` for that frame and is
  counted (`RealPerceptionEngine.pose_stats`); detection, tracking, risk and the frame status are unaffected.
- **Frozen contract kept.** `TrackObs.keypoints` (contract v1.0: exactly 17 `(x, y, conf)`, conf in [0, 1]) is filled;
  no contract change.

## 5. Adopted model: pretrained YOLO11s-pose

Official Ultralytics YOLO11s-pose, trained by Ultralytics on COCO keypoints, **unmodified**: 17 COCO keypoints, one
class (`person`), 9,918,238 parameters as loaded [M]. Licence: Ultralytics is AGPL-3.0 (already open decision OD-7 in
the project for the detector). Reasons for adoption are in §13.

## 6. Model weights, locations, checksums, sources

| Model | Role | Location | Size | sha256 / status | Source |
| --- | --- | --- | --- | --- | --- |
| YOLO11s-pose (pretrained, COCO) | **RUNTIME, integrated** | `models/yolo11s-pose.pt`; registry `pose_yolo11s_coco_v1` | 20,373,161 B | `1060bda4a27012060eca246f9b2adeea22eabb045a1e58f8d229be29b7ebc2ba` [M, re-verified 2026-11-27] | Ultralytics release v8.3.0 (GitHub assets) |
| YOLO11s-pose distilled (`yolo11s_pose_distill_960/best.pt`) | EXPERIMENTAL | `artifacts/pose/weights/experimental/yolo11s_pose_distill_960/best.pt` (from Drive `runs_pose/`) | 20,198,472 B | `0cc0896e8145d421d91a09fdef2e3291ed2f78a76ca59e28651f2e819c707a92` [M] | trained in Colab from the pretrained model |
| YOLO11s-pose E1 v1 (`yolo11s_pose_e1_ignore_960/best.pt`) | EXPERIMENTAL | `artifacts/pose/weights/experimental/yolo11s_pose_e1_ignore_960/best.pt` (from Drive `runs_pose/`) | 20,193,032 B | `23230614358e4e893ed54eafb21ac40ab63cf89399696e711cf9f28ae65efaa6` [M] | Colab |
| YOLO11s-pose E1b (`yolo11s_pose_e1b_centre_960/best.pt`) | EXPERIMENTAL | `artifacts/pose/weights/experimental/yolo11s_pose_e1b_centre_960/best.pt` (from Drive `runs_pose/`) | 20,193,032 B | `67c68e0b0e4d12e2ae6c77772dcbeeae6a1f1da92eacf0fa30b49d54eb99dddf` [M] | Colab |
| YOLO11s-pose E2 (`yolo11s_pose_e2_softconf_960/best.pt`) | EXPERIMENTAL | `artifacts/pose/weights/experimental/yolo11s_pose_e2_softconf_960/best.pt` (from Drive `runs_pose/`) | 20,193,032 B | `6fdd3a3eaa623550910eab7d96cd6913fef097ef7b69f9316c39edbdceb6a7d2` [M] | Colab |
| ViTPose-L SwimXYZ ONNX (`vitpose-l-swimxyz-mv.onnx`, 1.23 GB) | RESEARCH-ONLY teacher | Drive `MyDrive/pose_baseline/models/` | 1,177.2 MB (package log) | kept external (decision 2026-11-27) | Hugging Face `willalexm/vitpose-l-swimxyz`, exported to ONNX |

Full per-file sizes and checksums: `artifacts/pose/asset_inventory.csv`; upload guidance: `artifacts/pose/ASSET_MANIFEST.md`.

Experimental checkpoints are kept only as research artifacts; none is loaded by any configuration.

## 7. SwimXYZ and its relevance [H]

SwimXYZ (Fiche et al., MIG 2023; Zenodo, CC BY 4.0) is a **synthetic** swimming dataset: 11,520 videos (5 s, 60 fps,
3.4 M frames), 4 competitive strokes, 5 camera views, 2D/3D joints in COCO17 and other skeletons; 303.7 GB in total
(297 GB of video). It teaches swimming body shapes and views, but contains no real water, no real pool cameras, no
treading water, struggling or drowning behaviour (`docs/POSE_SWIMXYZ_STRATEGY.md`).

Its relevance here was indirect: the public **ViTPose-L fine-tuned on SwimXYZ** (`willalexm/vitpose-l-swimxyz`, 25,000
freestyle frames, 5 epochs, synthetic-validation PCK@0.05 0.982 per its model card) was used **offline as a teacher**
to produce pseudo-labels on our footage. ViTPose was never deployed, benchmarked on the PC, or used at runtime.

## 8. Dataset download status — what was actually obtained

| Data | Status |
| --- | --- |
| SwimXYZ (any part, 303.7 GB) | **Not downloaded** (assessed only; `docs/POSE_SWIMXYZ_STRATEGY.md`) |
| ViTPose-L SwimXYZ ONNX (`vitpose-l-swimxyz-mv.onnx`, 1,177.2 MB) | On Google Drive `MyDrive/pose_baseline/models/`; **kept external**, not in this package |
| Project frames `datasets (1).zip` | 5,613 frames, 1920×1080, one person box per frame [H, audit]; on Drive; **not** in this package (size/licence) |
| Pose subset `pose_annotation_set_v2.zip` (102.9 MB) | 362 frames: 300 `real_train`, 62 `real_val` (27 of them = test set); local copy in Downloads; only `manifest.csv` packaged |
| E3 hard set `e3_set` | 14 disputed held-out frames + manifest; packaged under `artifacts/pose/datasets_metadata/e3_set/` (images: see licence note in the manifest) |
| Experimental checkpoints, run outputs, E0–E3 / Test-27 evaluation tables, AI references | Retrieved from Drive as `pose_package_20261127.zip` (93.4 MB, 157 files, sha256 verified) and organised under `artifacts/pose/` |
| Human keypoint labels | **None exist.** All references below are AI-generated (§11) |

## 9. Training experiments and configurations [H]

All training ran in Google Colab (Ultralytics 8.4.220; Tesla T4, E2 on A100), starting from the COCO-pretrained
YOLO11s-pose; notebooks in `notebooks/`, scripts in `python_scripts/pose/`.

| Run (Drive `runs_pose/…`) | Data | Labels | Key settings | Notebook |
| --- | --- | --- | --- | --- |
| `yolo11s_pose_distill_960` (distillation) | 300 `real_train` frames; validation 35, test 27 (the 62 `real_val` frames) | ViTPose-L SwimXYZ predictions (AI pseudo-labels), uncertain points written as v=0 | imgsz 960, batch 16, T4; stopped at epoch 67 of 100 (patience 30 → best epoch 37) [M, `args.yaml`/`results.csv`] | `pose_distill_train.ipynb` |
| `yolo11s_pose_e1_ignore_960` (E1 v1) | 2,861 train / 762 val frames from 5,613 (held-out blocks 2, 13, 15, 16 excluded) | ViTPose filtered: conf ≥ 0.7, flip error ≤ 0.05, temporal jump ≤ 0.25; others "unknown" (v=1) | ignore-aware loss patch; imgsz 960, batch 16, 40/40 epochs, T4 (1.0 h) | `pose_e1_train.ipynb` |
| `yolo11s_pose_e1b_centre_960` (E1b) | same | same; unknown points placed at the box centre (fix for augmentation turning them into "absent") | same; augmentation check PASS | `pose_e1_train.ipynb` |
| `yolo11s_pose_e2_softconf_960` (E2) | same | positions from filtered points; confidence of every point trained towards the teacher's confidence | soft-confidence loss patch; 40 epochs, A100 (0.29 h) | `pose_e2_train.ipynb` |

Run outputs (exact `args.yaml`, per-epoch `results.csv`, curves, batch images) and the four `best.pt` checkpoints are
in `artifacts/pose/training/<run>/` and `artifacts/pose/weights/experimental/<run>/` [M, retrieved from Drive
2026-11-27, every file verified against the package's sha256 list]. Their validation metrics are agreement with each
run's own teacher-derived validation labels and are not comparable across runs.

Non-training diagnostics: **E0** (both models on all 5,613 frames plus flipped copies; agreement, flip consistency,
jitter; `pose_e0_diagnostics.ipynb`) and **E3** (evaluation on 14 most-disputed held-out frames;
`pose_e3_eval.ipynb`).

## 10. Distillation, E1, E1b, E2 — what happened [H]

- **Distillation**: improved agreement with the teacher on the 27 test frames (PCK@0.2 upper 0.40 → 0.62) but
  overfitted (val pose loss 9.3 → 8.3 while train fell 5.9 → 2.8) and **lowered lower-body confidence** (about
  0.75 → 0.45). Cause found: in Ultralytics a keypoint written `v=0` is trained as "absent", so every uncertain teacher
  point taught "this joint is not there".
- **E1 v1**: worse than distillation; confidences collapsed further (lower-body mean 0.24). Cause found: unknown points
  stored at (0, 0) were pushed outside the image by mosaic/affine augmentation and turned into "absent" again.
- **E1b** (fix): collapse gone but confidence **always high** (nose 0.79 though faces are rarely visible): no
  information; positions not better than distillation.
- **E2**: confidence level sane (faces low), but positions worse than distillation and confidence not more informative.
- Pre-registered success rules (fixed before each run) were **not met** by E1, E1b or E2. Training stopped.

## 11. Evaluation methodology and metrics

- **PCK@t**: fraction of labelled keypoints within *t* × the longer side of the person box (t = 0.05, 0.1, 0.2), for
  the upper body and all joints. **Swap-tolerant PCK**: one left/right decision per frame (taken by PCK@0.2), because
  back and inverted views make sides genuinely ambiguous.
- **Confidence AUROC**: does a model's keypoint confidence separate hits (within 0.1) from misses? 0.5 = no information.
- **References — none is human ground truth:**
  - *Teacher reference*: ViTPose-L SwimXYZ predictions on the 27 test frames (`ai_reference_unverified_test27.json`).
  - *E3 reference*: 14 hardest held-out frames, keypoints placed by an AI assistant (Claude) by visual review
    (`artifacts/pose_e3/pose_gt_e3.json`, provenance `ai_visual_review_unverified`).
  - *Independent AI visual reference*: 20 of the 27 test frames (7 too blurred), 201 points, placed by an AI assistant
    from gridded crops, independent of ViTPose and every YOLO model (`artifacts/pose_human_ref/pose_gt_t27_ai_visual.json`;
    the folder name is historical — the content is **not** human-labelled).
- All numbers are therefore **agreement with a reference, not accuracy**.

## 12. Comparison: pretrained vs experimental models [H]

Independent AI visual reference (20 frames, 201 points; differences under ~0.1 are noise):

| Metric | Pretrained | Distilled | E1b | E2 |
| --- | --- | --- | --- | --- |
| PCK@0.1 all / swap-tolerant | 0.423 / **0.522** | 0.328 / 0.453 | **0.443** / 0.498 | 0.428 / 0.483 |
| PCK@0.2 all / swap-tolerant | 0.632 / 0.726 | 0.632 / **0.771** | **0.701** / 0.751 | 0.662 / 0.711 |
| PCK@0.2 upper, swap-tolerant | 0.700 | **0.820** | 0.740 | 0.720 |
| Frames with no detection | 2 | 0 | 0 | 0 |
| Confidence AUROC | **0.770** | 0.670 | 0.694 | 0.677 |

Teacher-based sets favoured the distilled model (E3 swap-tolerant PCK@0.2 0.785 vs pretrained 0.608; Test-27 vs the
teacher 0.519 vs 0.376), but that partly measures **copying the teacher**. Against the independent reference the lead
largely disappears. Pretrained confidence AUROC was also best in E2's checks (0.75 E3 / 0.81 Test-27).

## 13. Reasons for retaining the pretrained model

1. No experimental model showed a clear, robust improvement against a reference independent of the teacher (§12).
2. The pretrained model's keypoint confidence is the most informative (highest AUROC), which matters for drawing only
   trustworthy joints and for any future confidence-gated features.
3. It is the official, reproducible, checksum-verifiable file; no custom training pipeline is needed at runtime.
4. The master blueprint's own rule: fine-tune only if pretrained keypoints fail — failure was not demonstrated.

## 14. Code integration details and modified files [M]

New: `src/lifeguard/ai/pose/` (`base.py`: `PoseEstimator` interface, crop / IoU / crop→frame helpers;
`ultralytics_pose.py`: batched crop inference, model-type check), `configs/ai_z5cc_pose.yaml`,
`tests/ai/test_pose.py`, `tests/ai/test_pose_real_model.py`, `tests/app/test_dashboard_pose.py`,
`python_scripts/pose/bench_pose_pipeline.py`, `python_scripts/pose/record_pose_demo.py`, `docs/POSE_INTEGRATION.md`.

Modified: `src/lifeguard/ai/config.py` (`PoseConfig`, off by default), `ai/perception/factory.py` (builds the
estimator; missing file → run without pose, checksum mismatch → refuse), `ai/perception/real_engine.py` (pose on
visible tracks, `pose_stats`, failure isolation, keypoint validation), `app/dashboard_controller.py` (skeleton overlay,
`pose` column), `app/runtime.py` (Components entry when enabled), `common/models.py` (registry kind `pose`),
`models/registry.yaml` (`pose_yolo11s_coco_v1`), `README.md`, `docs/DEMO.md`.

Configuration (`pose:` block): `enabled` (false), `weights`/`model`, `conf` 0.25, `imgsz` 320, `device`,
`crop_margin` 0.15, `min_iou` 0.3, `max_tracks` 8, `half` false, `allow_missing_weights` true. Enable with
`configs/ai_z5cc_pose.yaml`; disable with `configs/ai_z5cc.yaml` (identical output to before, tested).

## 15. Dashboard visualisation [M]

- Joint drawn only at confidence ≥ 0.5; radius grows with confidence; dark rim for contrast on water.
- Edge drawn only when both joints pass → no limb is ever drawn to a missing joint.
- Colours by body part, kept apart from the risk colours on boxes: **head white, torso teal, swimmer's left blue,
  swimmer's right amber**. The track-id label is drawn last so the skeleton never hides it.
- Swimmers table column `pose` = joints ≥ 0.5 out of 17 (`-` = no pose); Components panel lists the pose model with
  "display only; not used by the risk engine".

## 16. Demo videos and their results

Two layers, kept separate:

- **[H] Verified during integration (2026-11-27, earlier session)**: live dashboard runs of `demo_06` and `demo_01`
  (skeletons, pose column, Components entry visually checked; screenshots in
  `artifacts/pose/demo_results/dashboard_screenshots_integration_20261127/`), and pose OFF/ON benchmarks of 4 clips
  (`artifacts/pose/evaluation/integration_benchmarks_20261127/`).
- **[M] Reproduced in this task**: all 12 demo videos rerun through the full `LifeguardRuntime` (the dashboard's back
  end) with `configs/ai_z5cc_pose.yaml` + `configs/system_demo_drone.yaml`, each also with pose disabled for
  comparison (`python_scripts/pose/record_pose_demo.py`; outputs and overlays in `artifacts/pose/demo_results/`).

| Video | Frames | Errors | Tracks | Visible obs. with pose | PRE_ALERT / ALARM (video time) | Same events with pose off |
| --- | --- | --- | --- | --- | --- | --- |
| `demo_01_swim_I-EKqLdbysk` | 1232 | 0 | 7 | 1430 / 1873 (76 %) | ALARM 25.53 s (submersion timeout); ALARM 34.77 s (submersion timeout) | yes |
| `demo_02_drown_I-EKqLdbysk` | 308 | 0 | 7 | 575 / 1225 (47 %) | PRE_ALERT 7.43 s (detector distress) | yes |
| `demo_03_swim_L0KTqPloUiU` | 338 | 0 | 8 | 620 / 851 (73 %) | none | yes |
| `demo_04_drown_L0KTqPloUiU` | 307 | 0 | 9 | 880 / 988 (89 %) | PRE_ALERT 7.97 s (detector distress) | yes |
| `demo_05_swim_L0KTqPloUiU` | 367 | 0 | 5 | 243 / 375 (65 %) | none | yes |
| `demo_06_drown_T5mDQeDkca0` | 574 | 0 | 3 | 810 / 1158 (70 %) | PRE_ALERT 2.67 s (detector distress); PRE_ALERT 11.47 s (risk above threshold); ALARM 11.67 s (submersion timeout) | yes |
| `demo_07_swim_Y` | 904 | 0 | 11 | 635 / 1997 (32 %) | ALARM 26.0 s (submersion timeout) | yes |
| `demo_08_swim_Y` | 386 | 0 | 8 | 638 / 850 (75 %) | none | yes |
| `demo_09_drown_Y` | 382 | 0 | 4 | 63 / 255 (25 %) | none | yes |
| `demo_10_swim_Y` | 862 | 0 | 7 | 726 / 1526 (48 %) | none | yes |
| `demo_11_drown_iVMA7GhSJjM` | 632 | 0 | 2 | 315 / 701 (45 %) | PRE_ALERT 1.9 s (detector distress) | yes |
| `demo_12_drown_wQdAEA4a8IA` | 339 | 0 | 3 | 306 / 509 (60 %) | PRE_ALERT 5.47 s (detector distress) | yes |

Reading [M]: all 12 videos processed with 0 errors; every risk event (level, reason, time) was identical with pose on and off. PRE_ALERT (detector distress) fired on 5 of 6 near-drowning clips (02, 04, 06, 11, 12) at 1.9–8.0 s; `demo_09` (drowning starts mid-clip) raised nothing, as already documented in the demo guide. ALARM (submersion timeout) fired on `demo_06` (11.7 s, the swimmer goes under) and — **falsely** — on the normal-swimming clips `demo_01` (2 ALARMs) and `demo_07` (1 ALARM). Those false ALARMs are a **pre-existing limitation of the submersion rule** (swimmers leave the cropped 224 px frame), identical with pose disabled: not a pose regression. The demo guide's earlier note of the `demo_06` ALARM at about 17 s differs from the 11.7 s measured here; the cause of that difference was not investigated.

Screenshots: `artifacts/pose/demo_results/screenshots/` (3 frames per video with the most confident joints, rendered by the dashboard's own `annotate()`), and real dashboard captures in `artifacts/pose/demo_results/dashboard_screenshots_integration_20261127/`.

## 17. GTX 1650 performance [M]

Laptop: NVIDIA GTX 1650 4 GB, Intel i7-9750H, 16 GB RAM, Windows 10, Python 3.11, Ultralytics 8.3.40, FP32.

Rerun of the 12 demo videos (this task, full `LifeguardRuntime`, perception + risk + alerts per frame, no display; first 10 frames excluded):

| Video | Pose ON FPS | Pose OFF FPS | Pose ON latency mean / p95 (ms) | Pose ms / frame |
| --- | --- | --- | --- | --- |
| `demo_01_swim_I-EKqLdbysk` | 25.2 | 52.5 | 39.1 / 50.4 | 20.6 |
| `demo_02_drown_I-EKqLdbysk` | 19.0 | 49.5 | 51.9 / 65.3 | 32.0 |
| `demo_03_swim_L0KTqPloUiU` | 22.4 | 45.7 | 44.0 / 55.4 | 25.0 |
| `demo_04_drown_L0KTqPloUiU` | 20.2 | 47.2 | 48.9 / 59.8 | 29.7 |
| `demo_05_swim_L0KTqPloUiU` | 21.8 | 42.3 | 45.2 / 64.3 | 23.4 |
| `demo_06_drown_T5mDQeDkca0` | 20.4 | 46.9 | 48.3 / 69.8 | 26.2 |
| `demo_07_swim_Y` | 22.1 | 42.7 | 44.7 / 62.0 | 24.2 |
| `demo_08_swim_Y` | 19.3 | 46.4 | 51.1 / 75.7 | 28.0 |
| `demo_09_drown_Y` | 32.0 | 52.9 | 30.8 / 44.8 | 19.1 |
| `demo_10_swim_Y` | 24.3 | 50.3 | 40.5 / 50.7 | 21.2 |
| `demo_11_drown_iVMA7GhSJjM` | 24.8 | 49.5 | 39.8 / 57.3 | 20.1 |
| `demo_12_drown_wQdAEA4a8IA` | 25.2 | 50.7 | 39.1 / 48.0 | 20.8 |

Range: pose ON 19.0–32.0 FPS, pose OFF 42.3–52.9 FPS. The integration benchmark (`bench_pose_pipeline.py`, perception + risk engine only) gave 23–25 FPS on vs 46–57 FPS off (`docs/POSE_INTEGRATION.md` §5).

- Pose adds about 19–32 ms per frame (all crops of a frame in one batched call) and roughly halves processing FPS.
- GPU memory [H, integration benchmark]: torch peak allocation 89 MB → 164–174 MB; `nvidia-smi` 392 → 478–512 MB.
- Tried: FP16 42.7 ms (slower on this GPU); imgsz 256 no gain. Kept FP32 / 320.
- With pose on, processing (~23–25 FPS) is below the demo clips' 30 fps, so real-time playback lags slightly; it stays
  far above the health floor (5 FPS).

## 18. Safety constraints and current limitations

Constraints (enforced and tested):
- Pose is display/recording only; the risk engine never reads `TrackObs.keypoints`; there is no pose weight.
- Pose cannot raise WATCH, PRE_ALERT or ALARM; risk thresholds, persistence, hysteresis and FSM are unchanged.
- Detection and tracking logic unchanged; tracks and risk events were identical with pose ON vs OFF in every run.
- Pose failure cannot crash or disable detection, tracking or alerts.

Limitations:
- Real-pool pose accuracy is **unknown**: no sufficiently independent human-annotated validation set exists.
- Submerged, splashing or small swimmers often get few confident joints or none; legs are mostly invisible;
  left/right can swap on back and inverted views.
- Pose roughly halves processing FPS on the GTX 1650.
- The existing submersion rule raises false ALARMs on `demo_01` and `demo_07` (swimmers leaving the cropped 224 px
  frame); this is pre-existing and identical with pose off — not a pose regression.

## 19. Reproduction (local)

```powershell
cd E:\lifeguardDron\lifeguard-system
# 1. weights (skip if present; verifies sha256)
python -m lifeguard.common.models verify
# 2. tests
E:\lifeguardDron\.venv\Scripts\python.exe -m pytest -q
E:\lifeguardDron\.venv\Scripts\python.exe -m ruff check src tests
# 3. dashboard with pose (.venv-train has torch/Ultralytics; put this repo's src first)
$env:PYTHONPATH = "E:\lifeguardDron\lifeguard-system\src"
$env:LIFEGUARD_AI_CONFIG = "E:\lifeguardDron\lifeguard-system\configs\ai_z5cc_pose.yaml"
$env:LIFEGUARD_SYSTEM_CONFIG = "E:\lifeguardDron\lifeguard-system\configs\system_demo_drone.yaml"
& E:\lifeguardDron\.venv-train\Scripts\python.exe -m streamlit run src/lifeguard/app/dashboard.py
# 4. headless demo rerun and benchmark
& E:\lifeguardDron\.venv-train\Scripts\python.exe python_scripts/pose/record_pose_demo.py --videos E:\lifeguardDron\demo\demo_06_drown_T5mDQeDkca0.mp4 --out artifacts/pose/demo_results
& E:\lifeguardDron\.venv-train\Scripts\python.exe python_scripts/pose/bench_pose_pipeline.py --ai-config configs/ai_z5cc_pose.yaml --system-config configs/system.yaml --video E:\lifeguardDron\demo\demo_06_drown_T5mDQeDkca0.mp4
```

Training/evaluation notebooks (Colab, Drive `MyDrive/pose_baseline/`): `pose_audit`, `pose_zero_shot`,
`pose_baseline_eval`, `pose_distill_train`, `pose_e0_diagnostics`, `pose_e3_eval`, `pose_e1_train`, `pose_e2_train`,
`pose_t27_ai_visual_eval`, `pose_human_eval` (ready, never run: no human labels exist).

## 20. Future research recommendations [P]

1. **Human-labelled validation** (50–100 real-pool frames, two annotators on a subset for noise) — the only way to
   measure real accuracy; `notebooks/pose_human_eval.ipynb` is ready for it.
2. **Shadow mode**: compute left/right-invariant pose features (verticality, head height relative to the waterline,
   symmetric arm measures) on recorded clips, logged but not used, and test whether they separate distress from normal
   swimming better than box-and-motion features (master blueprint open question).
3. Only if (1) shows pretrained pose failing on the team's camera: targeted fine-tuning with human labels, not more
   teacher pseudo-labels.
4. Re-evaluate on the team's own calibrated camera; the demo clips are 224 px crops upscaled ×3.
5. Any use of pose in risk decisions requires a separate validated decision and an explicit configuration change.

## Related documents

`docs/POSE_INTEGRATION.md` (integration, tests, measurements) · `docs/POSE_SELF_SUPERVISED_RESEARCH.md` (E0–E2 and
reference checks, full tables) · `docs/POSE_SWIMXYZ_STRATEGY.md` · `docs/POSE_DATASET_DECISION_REPORT.md` ·
`docs/POSE_VITPOSE_INTEGRATION_ASSESSMENT.md` (ViTPose not approved for runtime) · `artifacts/pose/ASSET_MANIFEST.md`.
