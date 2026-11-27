# ViTPose-L SwimXYZ as the primary pose estimator: integration assessment

Date: 2026-11-27. Assessment only: nothing implemented, trained, downloaded or changed.
Legend: **[V]** verified (repository code/config, saved notebook outputs, model card) · **[A]** assumption or
estimate, not measured · **[R]** recommendation.

> **Decision (2026-11-27): not approved.** ViTPose-L will not be the runtime pose estimator, and the V1 benchmark
> (copying the ONNX to the PC) is not approved. The project continues with **YOLO11s-pose** (option B). This document
> is kept as the record of the assessment.

## 1. Executive recommendation

**Adopt architecture A (YOLO11 detection + ByteTrack + ViTPose-L on track crops) as the primary candidate, in
shadow mode only, conditional on one measurement we do not have yet: latency and memory on the GTX 1650.** Keep the
distilled YOLO11s-pose (B) as the planned fallback for crowded scenes, and pretrained YOLO-pose only as a
diagnostic (D).

Why:

- **ViTPose is the stronger estimator on our hardest footage** [V E3]. It has the best fine-tolerance agreement
  (PCK@0.1 all 0.615, swap-tolerant 0.654, against 0.354 / 0.469 for distilled YOLO), it is the only model whose
  confidence predicts agreement well (>= 0.9 confidence agreed 92 % of the time in E0), and it is the steadiest over
  time. These are agreement metrics against unverified AI references on 14 deliberately hard frames, not accuracy.
- **It fits the existing pipeline without changing it.** It is top-down: it takes the tracker's box and returns 17
  COCO keypoints for that track. Association with track IDs is therefore by construction, and the frozen
  `TrackObs.keypoints` contract (17 x (x, y, conf)) needs no change [V].
- **Its weakness is cost, and the cost grows with swimmer count.** It runs once per swimmer (36 ms per crop on a
  T4 [V]). The distilled YOLO runs once per frame for all swimmers (23 ms on a T4 at 960 px [V]). On the GTX 1650,
  ViTPose-L is estimated at roughly 80-120 ms per crop in FP32 [A]. That is real-time for 1 swimmer and needs
  scheduling (not every track every frame) beyond about 2 swimmers.
- **Not yet for production.** No human-labelled accuracy, no 1650 measurement and no multi-swimmer test exist. Pose
  stays at risk weight 0 until the validation plan (Section 7) is passed.

## 2. Proposed pipeline

```
frame ──► YOLO11 detector ──► ByteTrack ──► pool mapping / submersion timers / motion features   (unchanged)
                                   │
                                   └─► PoseScheduler: choose tracks due for a pose update this frame
                                           (TRACKED, in_water, not posed for >= 1/pose_hz s, max N per frame)
                                        ──► crop (box x 1.25, 4:3 aspect, affine 256x192) ──► batched ViTPose ONNX
                                        ──► decode heatmaps ──► image coords + confidence ──► per-track pose cache
TrackObs.keypoints  ◄── fresh cached pose for that track_id (or None if stale/missing)
pose features (later, L/R-invariant) ──► behaviour layer only, risk weight 0 (unchanged rule)
```

Detection, ByteTrack, submersion timers, the hard submersion rule and the risk FSM are untouched [R].

## 3. Technical integration requirements

| Area | Requirement | Status |
| --- | --- | --- |
| Model | `vitpose-l-swimxyz-mv.onnx`, 1.23 GB FP32, input `(N, 3, 256, 192)`, output `(N, 17, 64, 48)` heatmaps [V]. Apache-2.0 [V]. About 300 M parameters, from the file size [A]. | On Drive only, not on the PC |
| Preprocessing | Expand the box by 1.25, pad to 192:256, warp to 256x192, BGR to RGB, ImageNet mean/std. | Implemented and tested in `python_scripts/pose/pose_common.py` [V] |
| Decoding | Argmax per heatmap plus quarter-pixel refinement, back to image coordinates. Peak value as confidence, clamped to [0, 1] for the contract. | Implemented [V]. Decoding error about 2 px in a synthetic test [V]. |
| Contract | `TrackObs.keypoints`: exactly 17 COCO-ordered (x, y, conf), conf in [0, 1]; `FrameResult` checks points lie inside the frame [V]. Points outside the frame get conf 0. | No contract change |
| Track association | One crop per ByteTrack box gives one pose per `track_id` directly; no IoU matching needed (YOLO-pose would need it). | By design |
| Several swimmers | Each crop is independent; batch them (`N` dynamic in the ONNX graph [V]). | Supported |
| Engine hook | `RealPerceptionEngine._run` after `self._tracker.update` and before building `TrackObs` [V]; a failure must degrade to `keypoints=None`, never stop the engine. | New code (planned interface `PoseEstimator`) |
| Config | `AIConfig` uses `extra="forbid"` [V]: add an optional `pose:` block (model path or registry name, `pose_hz`, `max_tracks_per_frame`, `min_kpt_conf`, device, FP16). | New |
| Registry | `models/registry.yaml` kinds are `detector`, `mock_fixture` today [V]: add `kind: pose` with sha256. | New |
| Features | `feature_spec.json` lists `body_verticality` and `arm_pattern` as planned, blocked by pose keypoints [V]. Existing 8 features do not use keypoints [V]. | Later, minor spec version |
| Risk | `risk.weights.behavior: 0.0` [V]; detector distress can only reach PRE_ALERT [V]. Pose evidence must enter only through the behaviour layer. | Unchanged rule |

Confidence handling [R]: ViTPose peaks are heatmap values, not probabilities. E0 shows they are informative
(0.5-0.7: 39 %, 0.7-0.9: 67 %, >= 0.9: 92 % agreement with YOLO [V]). Use a per-keypoint threshold (start 0.5) and
mark the rest unknown; never fill gaps by interpolation across long intervals.

## 4. Performance and hardware

Measured on a Colab Tesla T4 [V]:

| Component | Latency | Unit |
| --- | --- | --- |
| ViTPose-L SwimXYZ ONNX, CUDA EP, FP32 | 33.6-36.9 ms | per swimmer crop, unbatched, including crop/normalise |
| YOLO11s-pose pretrained, 640 px | 12.2-12.4 ms | per **frame** (all swimmers) |
| YOLO11s-pose distilled, 960 px | 23.3-25.7 ms | per **frame** |
| E0 run: both models plus flipped copies, plus JPEG read | 117 ms | per frame |

On the GTX 1650 4 GB: the YOLO11s detector ran at about 14 ms per frame [V `PROJECT_PROGRESS_SUMMARY.md`]. The rest
is **not measured** [A]:

| Swimmers per frame | ViTPose every frame, FP32 (est. 80-120 ms per crop) | Distilled YOLO-pose (est. 40-70 ms per frame) |
| --- | --- | --- |
| 1 | ~95-135 ms total (7-10 fps): OK | ~55-85 ms: OK |
| 2 | ~175-255 ms: around the 5 fps floor | same as 1 |
| 5 | ~0.4-0.6 s: below the floor, needs scheduling | same as 1 |
| 10 | ~0.8-1.2 s: needs scheduling | same as 1 |

Notes:
- **Basis of the estimates [A].** The T4 has about 2.7x the FP32 throughput of the GTX 1650. Batching helps
  less than linearly on a small GPU.
- **The system budget [V].** `watchdog.fps_floor: 5.0` in `configs/system.yaml`: below 5 frames per second
  the perception status degrades. So pose must not block the frame loop.
- **Scheduling [R].** Run pose at, for example, 2-3 Hz per track, round-robin, with at most K crops per frame.
  Attach the cached pose with its age, and drop it after about 1 s. A pose refresh of 2-3 Hz is enough for
  posture features over a 4 s window [A].
- **FP16 [A].** Converting the ONNX to FP16 roughly halves memory (0.6 GB) and may give about 1.3-2x speed on
  Turing. Accuracy must be re-checked after conversion.
- **Memory [A].** FP32 weights 1.23 GB, plus CUDA/ORT workspace, plus the detector, should fit in 4 GB at batch
  sizes of about 8 or less. It must be measured.
- **Smaller crops are not an option.** The model is fixed at 256x192.
- **Future drone hardware** (for example Jetson-class) is unknown. ViTPose-L at about 300 M parameters is heavy for
  edge devices [A]. The distilled 10 M-parameter YOLO is far easier to deploy there, which keeps B relevant.

## 5. Risks and limitations

| Risk | Evidence | Mitigation |
| --- | --- | --- |
| Training domain | Fine-tuned on 25,000 **synthetic freestyle** frames, 5 epochs [V model card]; no real water, no treading or vertical distress | Real-footage validation (Section 7); the E3 sample is small |
| Head-down / inverted swimmers | Disputed frames in E0 were mostly head-down [V]; on these, the AI reference agreed more with ViTPose than with YOLO [V E3] | Include these orientations in the human-labelled set |
| Left/right on back and inverted views | 4 of 14 E3 frames better with L/R swapped for ViTPose, 10 of 14 for pretrained YOLO [V] | Make pose features L/R-invariant |
| Face points | Nose confident in only 53 % of frames; ears 41-47 % [V E0] | Do not build features on face points |
| Underwater limbs, splash, reflection | Lower confidence on the test-27 set (8.8 of 17 points >= 0.5) [V]; mirrored body above the surface seen in frame 47155 [V] | Confidence gating; ignore points above the waterline later (needs a waterline estimate) |
| Overlapping swimmers | Top-down crops can contain a second person; ViTPose returns one pose per crop and may mix bodies [A] | Multi-swimmer test; flag crops with high box overlap |
| Synthetic-to-real confidence | Peak values may be overconfident on unseen conditions [A] | Calibrate against human labels |
| Normal vs simulated distress | No distress footage exists; posture can look "abnormal" in normal play | Pose never decides; weight 0 until validated on real sessions |
| Latency | GTX 1650 not measured [A] | Benchmark first (V1) |
| Evaluation basis | All model comparisons so far are agreement metrics (ViTPose pseudo-labels, AI visual references) [V] | Human-labelled frames before any go decision |

## 6. Alternatives A-D

| | A. ViTPose-L on track crops | B. Distilled YOLO11s-pose | C. A, with B as fallback | D. ViTPose main, YOLO-pose diagnostic only |
| --- | --- | --- | --- | --- |
| Agreement on hard frames (E3, swap-tolerant PCK@0.1 / 0.2 all) | **0.654** / 0.762 | 0.469 / **0.785** | as A or B | as A |
| Cost scaling | linear in swimmers | constant per frame | switches | A plus occasional diagnostic runs |
| Track association | by construction | IoU matching needed | both paths | by construction |
| Confidence usefulness | good (E0) | pretrained YOLO poor (E0); distilled not measured | mixed | good |
| Edge/drone deployment | hard (about 300 M params) | easy (10 M) | flexible | hard |
| Complexity | low-medium (scheduler) | low | **high**: two estimators, switching logic, two validations | medium (diagnostics offline only) |
| Main failure mode | too slow with many swimmers; crop contamination | inherits teacher errors; weaker fine localisation | inconsistent features when switching models | as A |

[R] Choose **A** as the production candidate, with **D's** YOLO-pose as an *offline* diagnostic (agreement
monitoring on recorded sessions, never in the live path). Keep **B** as the documented fallback if the GTX 1650
benchmark fails. Avoid **C** for now: two live estimators double the validation work and make temporal features
switch character mid-track.

## 7. Validation plan (in order; each step gates the next)

| Step | What | Data | Needs | Output |
| --- | --- | --- | --- | --- |
| V1 | GTX 1650 benchmark: crops per second at batch 1/2/5/10, FP32 and FP16, peak GPU memory, alongside the detector | Recorded frames (no labels) | ONNX copied to the PC (1.23 GB from Drive; needs your approval), `onnxruntime-gpu` installed | latency table, memory, chosen `pose_hz` / batch |
| V2 | Human-labelled real-pool set: 60-100 frames across upright, horizontal, head-down, splash, occlusion, 2+ swimmers; 20 frames labelled twice for annotator noise | Held-out blocks plus new pool footage | **People** | first trusted PCK (normal and swap-tolerant) for A and B |
| V3 | Multi-swimmer: crop contamination and ID consistency | Clips with 2-10 swimmers | Footage with several swimmers | pose-to-track error rate |
| V4 | Temporal stability and missing-joint behaviour | Runs of near-consecutive frames (E0 tool) | none | jitter, dropout rate per keypoint |
| V5 | Shadow integration: pose computed and logged, `TrackObs.keypoints` filled, risk weight 0; test that pose alone cannot raise ALARM | Replayed sessions | code changes (later) | logs, unchanged alert behaviour |
| V6 | False pose interpretations: rate of "vertical/head-low" style pose indicators per hour on normal swimming | Normal sessions (Source B) | footage | false indicator rate |
| V7 | Go/no-go review | all above | team | decision |

Agreement metrics (E0, E3, ViTPose-vs-YOLO) are reported separately from V2's human-label accuracy throughout.

## 8. Go / no-go criteria (proposed, to be approved before V1)

**Go for shadow integration (V5)** if all hold:
1. V1: end-to-end processing >= 5 fps (the existing floor) with 1 swimmer at full pose rate, and >= 5 fps with 5
   swimmers at a per-track pose rate >= 2 Hz; peak GPU memory <= 3.2 GB.
2. V2 (human labels): ViTPose swap-tolerant PCK@0.2 >= 0.75 overall, and at least 0.10 above pretrained YOLO-pose.
   Upper-body points confident (>= 0.5) in >= 80 % of in-water frames.
3. V2: on head-down/inverted frames, ViTPose is not worse than distilled YOLO at PCK@0.1, swap-tolerant.

**Go for any non-zero risk weight** additionally requires V3, V4 and V6 results, plus a validated temporal feature
that lowers false alerts or time-to-alert on real sessions. That decision is outside this assessment.

**No-go / switch to B** if criterion 1 fails even with FP16 and scheduling, or if V2 shows distilled YOLO within 0.05
of ViTPose at PCK@0.1 swap-tolerant.

## 9. Sources

- Repository [V]: `src/lifeguard/contracts/types.py` (TrackObs keypoints), `src/lifeguard/ai/perception/real_engine.py`
  (frame loop), `src/lifeguard/ai/config.py` (AIConfig, `extra="forbid"`), `src/lifeguard/ai/features/feature_spec.json`,
  `models/registry.yaml`, `configs/system.yaml` (risk weights, `fps_floor`).
- Experiments [V]: `docs/POSE_SELF_SUPERVISED_RESEARCH.md` sections 8-9 (E0, E3), notebooks `pose_baseline_eval`,
  `pose_distill_train`, `pose_e0_diagnostics`, `pose_e3_eval`.
- Model card [V]: <https://huggingface.co/willalexm/vitpose-l-swimxyz>.
