# Pose integration (pretrained YOLO11s-pose) — display and recording only

**Status (2026-11-27):** integrated, tested, and run in the dashboard on the demo clips. Pose keypoints are
**shown and recorded; they do not contribute to any risk decision.** The risk engine does not read
`TrackObs.keypoints`, there is no pose weight, and pose can never raise WATCH, PRE_ALERT or ALARM.

## 1. Model

| Item | Value |
| --- | --- |
| Model | Official Ultralytics **YOLO11s-pose**, COCO-pretrained, unmodified (17 COCO keypoints, class `person`, 9.9 M parameters) |
| Source | `https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-pose.pt` (downloaded 2026-11-27) |
| File | `models/yolo11s-pose.pt` (git-ignored, 20,373,161 bytes) |
| sha256 | `1060bda4a27012060eca246f9b2adeea22eabb045a1e58f8d229be29b7ebc2ba` |
| Registry entry | `pose_yolo11s_coco_v1` (`kind: pose`) in `models/registry.yaml` |
| Reported as | `FrameResult.model_versions["pose"] = "yolo11s-pose.pt:sha256:1060bda4a270"` |

Why this model: the pose experiments (`docs/POSE_SELF_SUPERVISED_RESEARCH.md`, sections 8–10) found no reliable
advantage for the distilled, E1/E1b or E2 checkpoints over the pretrained model against an independent reference,
and the pretrained model had the most informative keypoint confidence. Those experimental checkpoints and ViTPose
are **not** used anywhere in the system. Re-download and verify: `python -m lifeguard.common.models fetch
pose_yolo11s_coco_v1`.

## 2. Design

```
frame ─► person detector (Z5cc) ─► ByteTrack ─► visible tracked boxes ─► crops (box + 15 % margin)
                                        │                                   │ one batched predict
                                        ▼                                   ▼
                                  TrackObs (ids, boxes,          YOLO11s-pose per crop → person in the crop that
                                  features, submersion)  ◄────── overlaps the tracked box best (IoU ≥ 0.3)
                                        │                        → keypoints shifted to frame pixels
                                        ▼                          → TrackObs.keypoints (same track id)
                                  risk engine (unchanged; ignores keypoints)
```

- The pose model never creates or removes tracks. It runs only on crops of boxes ByteTrack reports as `tracked`
  (no full-frame second detector). LOST tracks (predicted boxes) get no pose.
- Association: a crop can contain a neighbour; the pose person box that overlaps the tracked box most is kept, or
  none if the best IoU is below `min_iou` → `keypoints = None`.
- Coordinates: crop pixels + crop origin → frame pixels. A joint that is not detected, not finite or falls outside the
  frame is reported as `(0, 0, 0)`; joints are never moved or invented. Confidence is clipped to [0, 1].
- Contract: `TrackObs.keypoints` (frozen v1.0, 17 × (x, y, conf)) is filled; no contract change.
- Failure isolation: any pose exception (or malformed output) leaves `keypoints = None` for that frame, is counted
  in `RealPerceptionEngine.pose_stats.errors` and logged (first, then every 100th); the frame stays `status=ok`.
- Code: `src/lifeguard/ai/pose/` (`base.py` interface and geometry, `ultralytics_pose.py`), wiring in
  `ai/perception/real_engine.py` and `ai/perception/factory.py`, config `PoseConfig` in `ai/config.py`.

## 3. Configuration

`configs/ai_z5cc_pose.yaml` = `configs/ai_z5cc.yaml` (unchanged detector settings) + this block:

| Key | Default | Meaning |
| --- | --- | --- |
| `pose.enabled` | `false` | Off unless set; `configs/ai_z5cc.yaml` and every other config keep pose off |
| `pose.weights` / `pose.model` | — | Exactly one: a file path (verified against the registry) or a registry name |
| `pose.conf` | 0.25 | Pose model's person confidence inside a crop |
| `pose.imgsz` | 320 | Crop inference size |
| `pose.device` | null | `"0"`, `"cpu"` or Ultralytics default |
| `pose.crop_margin` | 0.15 | Crop enlargement (fraction of the box's longer side) |
| `pose.min_iou` | 0.3 | Pose person box vs tracked box for association |
| `pose.max_tracks` | 8 | Tracks per frame that get a pose (highest detector confidence first) |
| `pose.half` | false | FP16; **slower** on the GTX 1650 (section 5) |
| `pose.allow_missing_weights` | true | Missing **file** → warning, run without pose (`model_versions["pose"] = "unavailable: weights file missing"`). `false` → refuse to start. A registry **checksum mismatch** or wrong model type always refuses to start |

Enable / disable: use `configs/ai_z5cc_pose.yaml` (on) or `configs/ai_z5cc.yaml` (off), or set `pose.enabled`.
With pose off the engine output is identical to before (tested).

Dashboard: skeletons are drawn on tracks with keypoints. A joint is drawn only at confidence ≥ 0.5, larger when more
confident, with a dark rim for contrast on water; an edge is drawn only when both of its joints pass, so no limb is
drawn to a missing joint. Colours (kept apart from the risk colours on the boxes): head white, torso teal, swimmer's
left arm/leg blue, right arm/leg amber. The track id label is drawn last. The Swimmers table has a `pose` column
(`k/17` joints ≥ 0.5, `-` = no pose); the Components panel lists the pose model with "display only".

## 4. Run the demo

`E:\lifeguardDron\.venv` has no Ultralytics/torch, so the real pipeline cannot run there. `.venv-train` has them,
but its installed `lifeguard` package points to another checkout (`E:\lifeguard-github-upload`); put this
repository's `src` first:

```powershell
cd E:\lifeguardDron\lifeguard-system
$env:PYTHONPATH = "E:\lifeguardDron\lifeguard-system\src"
$env:LIFEGUARD_AI_CONFIG = "E:\lifeguardDron\lifeguard-system\configs\ai_z5cc_pose.yaml"
$env:LIFEGUARD_SYSTEM_CONFIG = "E:\lifeguardDron\lifeguard-system\configs\system_demo_drone.yaml"
& E:\lifeguardDron\.venv-train\Scripts\python.exe -m streamlit run src/lifeguard/app/dashboard.py
```

Sidebar: Input **video**, Video file e.g. `E:\lifeguardDron\demo\demo_01_swim_I-EKqLdbysk.mp4` (press Enter), **Start**.
Check: Components shows `pose: yolo11s-pose.pt:sha256:1060bda4a270 (display only; ...)`.

## 5. Measured on this laptop (GTX 1650 4 GB, i7-9750H, Windows 10, 2026-11-27)

`python_scripts/pose/bench_pose_pipeline.py` runs the same frames with pose OFF and ON through the real engine and
the real risk engine (Ultralytics 8.3.40, torch CUDA, FP32, pose imgsz 320; first 10 frames excluded from timing).
Processing FPS = perception + risk per frame, without display. Demo clips are 672 × 672 at 30 fps.

| Clip | Frames | Max swimmers / frame | OFF: FPS (ms) | ON: FPS (mean / p95 ms) | Pose ms / frame | Visible track obs. | With pose |
| --- | --- | --- | --- | --- | --- | --- | --- |
| demo_06 (drown) | 574 | 3 | 56.5 (17.7) | 24.8 (40.3 / 50.2) | 20.9 | 1158 | 810 (70 %) |
| demo_05 (swim) | 367 | 3 | 55.5 (18.0) | 24.1 (41.6 / 63.3) | 21.6 | 375 | 243 (65 %) |
| demo_11 (drown) | 600 | 2 | 55.3 (18.1) | 25.5 (39.2 / 49.0) | 19.7 | 669 | 310 (46 %) |
| demo_01 (swim, first 600) | 600 | 4 | 50.0 (20.0) | 23.1 (43.3 / 51.1) | 23.4 | 785 | 490 (62 %) |
| demo_01 (full, demo system config) | 1232 | 4 | 45.8 | 23.2 | 22.3 | 1873 | 1430 (76 %) |

- GPU memory: torch peak allocation 89 MB → 164–174 MB with pose; `nvidia-smi` used memory 392 MB → 478–512 MB
  (whole GPU, after the run).
- Optimisation tried: crops are already batched in one call. FP16: 42.7 ms (slower on this GPU). imgsz 256: 20.9 ms
  (no gain; per-call overhead dominates for small crops). Kept FP32 / 320.
- Consequence: with pose on, processing (~23–25 FPS) is below the 30 fps of the demo clips, so real-time playback
  runs slightly behind the video clock. It stays far above the health floor (`fps_floor: 5`).
- Missing keypoints (demo_06, pose observations): share of joints at confidence ≥ 0.5 per joint is 13–29 %
  (shoulders 0.29, ankles 0.15–0.16, ears 0.13–0.18). Most joints of submerged or blurred swimmers are below the
  draw threshold and are not drawn, which is intended.
- No pose errors and no frame errors in any run.

**Behaviour unchanged (verified on all runs above):** every track (id, box, state, features, submersion) and every
risk event (level, reason, risk, time) was identical with pose OFF and ON. Examples: demo_06 NORMAL→WATCH→PRE_ALERT
→ALARM (submersion timeout) at the same times; demo_11 with `system_demo_drone.yaml` PRE_ALERT (detector distress) at
1.9 s in both runs; demo_01 (a swim clip) raises 2 submersion-timeout ALARMs in **both** runs — a pre-existing
limitation of the submersion rule on that clip, not caused by pose.

## 6. Tests (2026-11-27)

- `E:\lifeguardDron\.venv` (no Ultralytics): `pytest` **531 passed, 2 skipped** (real-model test skipped: no
  ultralytics; GUI calibration test skipped as before). Baseline before the change: 509 passed, 1 skipped.
  `ruff check src tests` clean; `generate_fixtures --check` exit 0; `replay_mock` exit 0.
- `E:\lifeguardDron\.venv-train` (Ultralytics + CUDA; pytest made importable from a scratch copy, nothing installed):
  `tests/ai/test_pose_real_model.py`, `tests/ai/test_pose.py`, `tests/app/test_dashboard_pose.py`,
  `tests/ai/test_ultralytics_detector.py` — **27 passed**, including loading/verifying the official weights and
  rejecting a detector checkpoint as a pose model.
- New tests cover: crop and crop→frame transforms; association to the right swimmer when a neighbour is in the
  crop; batching and `max_tracks`; keypoints attached to the correct track id with two swimmers; LOST tracks get no
  pose; estimator exceptions and malformed output never change tracks or status; pose disabled = previous output
  exactly; pose cannot trigger or change PRE_ALERT/ALARM (steady swimmer with full-confidence joints, and a real
  submersion escalation identical with/without pose); the risk engine ignores keypoints; config validation; missing
  weights fallback vs refusal; checksum mismatch refusal; registry entry; overlay (confident joints drawn, low/missing
  joints and their limbs not drawn, display scaling, switch-off); Components entry only when pose is enabled.

## 7. Limitations

- Not validated as a pose reference on pool video: accuracy is unknown; agreement studies with AI references gave
  roughly half of the points within 10 % of the body size (`docs/POSE_SELF_SUPERVISED_RESEARCH.md`).
- Submerged, splashing or small swimmers often get few confident joints or none; left/right can swap.
- On the GTX 1650 pose roughly halves processing FPS (section 5).
- Pose is evidence for display and recording only. Using it in risk decisions needs a separate, validated decision
  (human-labelled data, shadow-mode evaluation) and an explicit configuration change; none exists.

## 8. Demo rerun of all 12 videos (2026-11-27, after integration)

`python_scripts/pose/record_pose_demo.py` ran every `E:\lifeguardDron\demo\demo_NN_*.mp4` through the full
`LifeguardRuntime` (the dashboard's back end) with `configs/ai_z5cc_pose.yaml` + `configs/system_demo_drone.yaml`, and
again with pose disabled. Results: 0 frame errors on all 12; risk events identical with pose on and off on all 12;
PRE_ALERT on 5 of 6 near-drowning clips (1.9–8.0 s; `demo_09` none, as documented before); ALARM on `demo_06`
(11.7 s) and pre-existing false submersion ALARMs on `demo_01` (2) and `demo_07` (1), identical with pose off;
processing 19.0–32.0 FPS with pose vs 42.3–52.9 FPS without (runtime incl. alerts, no display). Per-video table,
overlays and dashboard captures: `artifacts/pose/demo_results/`; summary: `docs/POSE_ESTIMATION_BLUEPRINT.md` §16–17.

## 9. Related assets

Runtime weight: `models/yolo11s-pose.pt` only. Experimental checkpoints (distillation, E1, E1b, E2), their training
outputs, evaluation tables and AI references are research artifacts under `artifacts/pose/` (never loaded by any
configuration); see `artifacts/pose/ASSET_MANIFEST.md`.
