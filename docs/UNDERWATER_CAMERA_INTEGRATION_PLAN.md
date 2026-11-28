# Underwater secondary camera — integration plan

Date: 2026-11-28 · Status: **planning document only**. Nothing described as "proposed" exists in the code; nothing
about the underwater camera is implemented or validated.

Labels used throughout: **[Verified]** checked in the repository (file given) · **[Proposed]** a design suggestion ·
**[Assumption]** believed but not checked · **[Not yet validated]** exists or is plausible but untested for this use ·
**[TBD]** needs evidence or a decision.

---

## A. Executive summary

**Why.** The system watches a pool from one above-water camera. Its strongest alarm, the submersion rule, works by
noticing that a swimmer in the pool *stopped being visible* and did not come back within `t_sub_s` (10 s by default)
[Verified: [`submersion.py`](../src/lifeguard/ai/features/submersion.py),
[`risk/config.py`](../src/lifeguard/system/risk/config.py)]. Above-water cameras cannot see a submerged swimmer; an
underwater camera can. The master Engineering Blueprint already lists "Underwater cameras: directly observe
submerged swimmers" and "Multi-camera fusion: several calibrated cameras mapped to one pool coordinate frame" as
future upgrades (§14, "Advanced Version / Future Improvements") [Verified: `Engineering Blueprint5.pdf`, p. 58].

**Complementary, not a replacement.** The above-water pipeline stays the primary system, unchanged. The underwater
camera adds a second view whose evidence is first logged in shadow mode and only later, after validation and explicit
approval, may contribute to decisions.

**What already exists [Verified].** Live-camera ingest by index or stream URL with reconnect; a real perception engine
(YOLO detector → ByteTrack → pool mapping → submersion timers → features → optional pose); a rule-based risk engine
with a per-track state machine; incident and alert handling; a watchdog; an operator dashboard. Camera identity is
already part of the data contract (`cam_id`, `track_uid = cam_id:session_id:track_id`).

**What needs investigation first.** The runtime and dashboard handle **one camera per instance**: running two
cameras at once, giving each its own calibration, aligning their timestamps, associating a swimmer across the two
views and deciding how underwater evidence may affect risk are all open (sections C, D, G, H).

**Models first, training last.** The existing detectors (Z5cc, ddt) and the pretrained YOLO11s-pose must be run on
real footage from the actual underwater camera and measured before anything else. Fine-tuning is a conditional
fallback if — and only if — that evaluation shows a gap that non-training changes cannot close. It is **not** an
approved or planned task.

## B. Existing system compatibility

| Component (file) | Reusable as-is | Needs camera-specific configuration | Needs architectural extension | Needs validation |
| --- | --- | --- | --- | --- |
| Live ingest `LiveCameraSource` ([`ingest/live.py`](../src/lifeguard/system/ingest/live.py)) — index or URL, latest-frame policy, reconnect every 2 s | Yes [Verified] | Camera index / stream URL | No (one instance per camera) | Underwater camera's stream format and latency [TBD] |
| Recorded video `VideoFileSource` ([`ingest/video.py`](../src/lifeguard/system/ingest/video.py)) | Yes — for offline evaluation of underwater recordings | — | No | — |
| Perception engine `RealPerceptionEngine` ([`real_engine.py`](../src/lifeguard/ai/perception/real_engine.py)) | Yes, per camera instance | Detector weights, `imgsz`, class map, tracker settings | No for independent use | Whole chain on underwater footage |
| Detectors Z5cc / ddt ([`ultralytics_detector.py`](../src/lifeguard/ai/detection/ultralytics_detector.py), [`configs/ai_z5cc.yaml`](../configs/ai_z5cc.yaml), [`configs/ai_ddt.yaml`](../configs/ai_ddt.yaml)) | Technically yes | Config per camera | No | **Yes — different visual domain** (section E) |
| ByteTrack ([`tracking/bytetrack.py`](../src/lifeguard/ai/tracking/bytetrack.py)) | Yes; one tracker per engine, IDs independent per camera | Possibly thresholds [TBD] | No | Continuity under bubbles/occlusion |
| Pose YOLO11s-pose ([`ai/pose/`](../src/lifeguard/ai/pose/), [`configs/ai_z5cc_pose.yaml`](../configs/ai_z5cc_pose.yaml)) | Yes (display only) | `pose:` block per camera | No | Yes (section F) |
| Pool geometry ([`calibrated.py`](../src/lifeguard/system/geometry/calibrated.py)) | The file format can hold a calibration per camera | A separate calibration file | **Yes** — the model is a single water-surface plane (section G) | Yes |
| Submersion monitor and risk engine ([`submersion.py`](../src/lifeguard/ai/features/submersion.py), [`risk/engine.py`](../src/lifeguard/system/risk/engine.py), [`risk/evidence.py`](../src/lifeguard/system/risk/evidence.py)) | **Not for the underwater view** (signals assume an above-water view, section H) | — | **Yes** — camera-aware evidence | Yes |
| Runtime `LifeguardRuntime` ([`runtime.py`](../src/lifeguard/app/runtime.py)) | One instance per camera only | `system.yaml` per camera (`cam_id`, geometry, response) | **Yes** for a single multi-camera runtime | Yes |
| Dashboard ([`dashboard.py`](../src/lifeguard/app/dashboard.py), [`dashboard_controller.py`](../src/lifeguard/app/dashboard_controller.py)) | One view per server | — | **Yes** for two views in one dashboard | Yes |
| Alerts, incident store, watchdog ([`alerts/`](../src/lifeguard/system/alerts/), [`health/watchdog.py`](../src/lifeguard/system/health/watchdog.py)) | Yes per instance; the store records `cam_id` | Separate database per instance, or a shared one [TBD] | Yes for a combined operator view | Yes |

Accepting a camera index or stream URL does **not** mean two cameras can run together; section C shows that the
runtime is single-camera.

## C. Multi-camera architecture investigation

### C.1 What the code does today [Verified]

- `SystemConfig` holds **one** `cam_id` (default `cam_sim`) and **one** `geometry.calibration`
  ([`system/config.py`](../src/lifeguard/system/config.py), [`geometry/config.py`](../src/lifeguard/system/geometry/config.py)).
- `LifeguardRuntime.__init__` builds exactly one geometry, one perception engine, one risk engine, one incident store
  and alert service, one watchdog, one clip recorder and one response platform; `process(frame)` and `run(source)`
  consume **one** frame stream ([`runtime.py`](../src/lifeguard/app/runtime.py)).
- The dashboard keeps a single `DashboardController` (`st.cache_resource`, "one runtime is shared by every browser
  tab of this server"), selects one input and shows one image ([`dashboard.py`](../src/lifeguard/app/dashboard.py)).
- Camera identity already flows through the contracts: `FrameResult.cam_id`, `track_uid = cam_id:session_id:track_id`
  ([`contracts/types.py`](../src/lifeguard/contracts/types.py)); the risk engine stores `cam_id` per track and times out
  only tracks whose uid has the current frame's `cam_id:session_id:` prefix
  ([`risk/engine.py`](../src/lifeguard/system/risk/engine.py), `_handle_absent`); `SubmersionMonitor` is documented as
  "per-camera"; the incident table has a `cam_id` column ([`alerts/store.py`](../src/lifeguard/system/alerts/store.py)).
- Live timestamps: `t_capture` is the **host** `time.monotonic()` when the reader thread receives the frame, `t_utc` the
  host clock, `t_media` is `None` for live cameras ([`ingest/live.py`](../src/lifeguard/system/ingest/live.py)). There is no
  camera hardware timestamp and no cross-camera synchronisation.

**Conclusion [Verified]:** the system does **not** support simultaneous cameras today. The contracts are camera-aware,
which makes extension easier, but no code path ingests, processes, displays or fuses two streams together.

### C.2 Options [Proposed]

| Option | Description | Pros | Cons / risks |
| --- | --- | --- | --- |
| **1. Two independent runtimes** (separate processes, separate `system.yaml` with distinct `cam_id`, separate dashboard ports) | No code change; underwater instance with `response.platform: none` and its own incident database | Fastest way to collect underwater evidence; above-water instance untouched | Two operator screens, two incident logs; no association or fusion; the underwater instance would run the above-water risk rules, which are not valid for that view (H.2) — so it must be shadow/observation only. Running two instances side by side is **[Not yet validated]** (GPU memory and FPS on the GTX 1650, port and file conflicts) |
| **2. One runtime, N perception engines** | One process, one engine per camera, a shared risk/alert layer | One operator view; the place where fusion would live | Architectural change in `runtime.py`, the dashboard and the watchdog; must keep the above-water path behaviourally identical |
| **3. Option 1 first, then 2** | Use option 1 for data collection and shadow evaluation; design option 2 only after evidence | Lowest risk; decisions based on measurements | Slower to a single operator view |

Recommendation [Proposed]: **option 3**.

### C.3 Requirements for simultaneous operation [Proposed]

| Topic | Requirement |
| --- | --- |
| Camera inputs and identity | One source per camera; a distinct `cam_id` (e.g. `cam_above`, `cam_under` [TBD names]); camera metadata (model, resolution, fps, mount position) recorded with every session |
| Ingestion | Independent reader threads (the latest-frame policy already drops stale frames per source) |
| Detection and tracking | One detector + ByteTrack per camera; **track IDs are independent per camera** and must never be compared across cameras |
| Timestamps | Both streams stamped on one host clock (already true for one process; must be preserved across processes [TBD]); measure each camera's capture-to-receipt latency, because host receipt time ≠ capture time |
| Alignment | Cross-camera comparisons use time windows (not exact frame pairs); tolerance [TBD from measured latency and frame rates] |
| Different fps / resolution | Each engine keeps its own frame size (the engine rejects size changes per instance); no assumption of equal fps |
| Disconnection | Per-camera watchdog faults; loss of the underwater camera must never stop or degrade the above-water pipeline (H.4) |
| Delay / dropped frames | Per-camera FPS and latency monitoring; underwater evidence older than a limit [TBD] is ignored |
| Compute | Measure both pipelines together on the target hardware; the above-water pipeline's FPS must not fall below its current level [TBD acceptance] |
| Dashboard | Two views (side by side or switchable), each labelled with its camera; underwater evidence clearly marked "shadow / not used for alerts" until approved |

## D. Cross-camera association and evidence fusion

Three separate capabilities — none exists today [Verified: no such code in `src/`]:

| Capability | Meaning | Status |
| --- | --- | --- |
| Independent per-camera tracking | ByteTrack IDs within one camera | Exists per camera instance [Verified] |
| Cross-camera swimmer association | Deciding that above-water track A and underwater track B are the same person | **Not implemented** |
| Multi-camera evidence fusion | Combining evidence from both cameras into one risk assessment per person | **Not implemented** |

Possible association approaches [Proposed], with limitations:

| Approach | Idea | Limitations |
| --- | --- | --- |
| Shared pool coordinates | Map both cameras to the same pool metres (C-18 frame) and associate by position within a time window | The current mapping projects the box centre onto the **water-surface plane**; an underwater body is not on that plane, so positions are biased by depth and refraction (section G). Accuracy unknown [TBD] |
| Temporal co-occurrence | A swimmer disappearing above water while one appears or stays near the same position underwater | Ambiguous with several swimmers nearby; depends on latency alignment |
| Motion patterns | Compare trajectories (direction, speed) over a window | Underwater and surface views see different body parts; weak for still swimmers — exactly the case of interest |
| Appearance | Colours of swimwear or caps | Water absorbs colour with depth, refraction and lighting change appearance; caps may be hidden underwater [Assumption] |

When association is uncertain, the system must treat the observations as **unassociated** and never merge risk
histories or transfer incidents (section H.4). The bar for any association rule is measured on recorded sessions with
human-reviewed identities [TBD].

## E. Existing detector suitability

| Model | Training domain [Verified: [`MODEL_TRAINING_REPORT.md`](MODEL_TRAINING_REPORT.md)] | Underwater relevance |
| --- | --- | --- |
| Z5cc YOLO11s (`models/z5cc_s448_final.pt`) | Side-view pool camera; 224 × 224 CVAT-annotated YouTube pool clips; held-out test mAP50 0.704 | Whether any training clips are underwater is not stated in the report [Unverified]. Underwater footage is a different visual domain [Assumption, to be measured] |
| ddt YOLO11s (`models/ddt_v2_final.pt`) | Overhead / high camera; held-out test mAP50 0.636 | Overhead ≠ underwater; likely the weaker starting point [Assumption] |

A missed detection removes the swimmer from tracking, submersion logic and pose: detector recall is therefore the
first thing to measure.

Initial strategy [Proposed]: run both existing detectors unchanged on recordings from the actual underwater camera
(offline, with [`run_video`](../src/lifeguard/app/run_video.py) or the headless recorder
[`record_pose_demo.py`](../python_scripts/pose/record_pose_demo.py)) and measure, on a human-annotated evaluation set:

- detection precision and recall (and per-frame recall at the operating confidence `conf: 0.1` used for ByteTrack);
- false positives (reflections, lane ropes, shadows, equipment);
- missed swimmers, split by situation (near/far, near the floor, under splash/bubbles, partially out of frame);
- performance under occlusion (swimmers overlapping);
- detection stability across consecutive frames;
- ByteTrack continuity: ID switches, fragmentations, track lifetime vs ground-truth presence.

Retraining is not recommended by default; see section J, stage 3 exit criteria.

## F. Pose estimation suitability

Pose is the component with the most underwater evidence so far — but it is **agreement with AI references, not
verified accuracy**, and it remains **display-only**.

### F.1 Verified evidence

- The 5,613 frames used in the pose work (1920 × 1080, one person box each) come from **one underwater side-view
  camera** [Verified: visual inspection recorded in [`POSE_SELF_SUPERVISED_RESEARCH.md`](POSE_SELF_SUPERVISED_RESEARCH.md)
  §3 and [`POSE_SWIMXYZ_STRATEGY.md`](POSE_SWIMXYZ_STRATEGY.md); audit
  [`AUDIT_REPORT.json`](../artifacts/pose/datasets_metadata/audit/AUDIT_REPORT.json)]. It is a different camera from the
  one the team will install.
- **E0, all 5,613 frames — model-vs-model agreement, no reference** [Verified:
  [`e0_report.txt`](../artifacts/pose/evaluation/drive/e0/e0_report.txt), research doc §8]: the pretrained YOLO11s-pose
  found no person matching the dataset box in **788 frames (14 %)**; agreement with ViTPose-L SwimXYZ within 0.1 of the
  box size was 72–81 % for hips/knees/ankles, 64–67 % shoulders/wrists, 55 % elbows, 64–75 % face; consensus coverage
  was 0.36 for the upper body and 0.47 for the lower body. Two models agreeing is not proof that either is correct.
- **Independent AI visual reference, 20 test frames, 201 points** (keypoints placed by an AI assistant from zoomed
  crops; **not human ground truth**) [Verified:
  [`t27vis_pretrained.csv`](../artifacts/pose/evaluation/drive/human/t27vis_pretrained.csv), research doc §10]. PCK@t =
  share of reference keypoints within *t* × the longer side of the person box
  ([`pose_common.py`](../python_scripts/pose/pose_common.py), `pck`); swap-tolerant = one left/right decision per frame.
  For the **adopted pretrained model**:

  | Metric | All joints | All, swap-tolerant | Upper body, swap-tolerant |
  | --- | --- | --- | --- |
  | PCK@0.1 | 0.423 | 0.522 | 0.51 |
  | PCK@0.2 | 0.632 | 0.726 | 0.70 |
  | Frames without a matched person | 2 of 20 | | |

**Correction of an earlier statement.** A previous summary said the model "placed about half the joints within 10 % and
70–82 % within 20 %" on "the 5,613 frames". Precisely: those figures come from the **20-frame** AI visual reference, not
from the 5,613 frames (where only model-to-model agreement was measured). "About half within 10 %" matches the
pretrained model (0.42 all, 0.52 swap-tolerant). "70–82 % within 20 %" was the swap-tolerant range across **all four**
models (upper body 0.70–0.82, all joints 0.71–0.77); for the pretrained model the values are 0.63 (all joints),
0.73 (swap-tolerant) and 0.70 (upper body, swap-tolerant).

### F.2 Considerations for the new camera [Assumption unless marked]

| Factor | Expected effect |
| --- | --- |
| Upper body | Visible in side and below views; arms move fast and blur |
| Legs | In the E0 underwater footage lower-body consensus (0.47) exceeded upper-body (0.36) [Verified, agreement only] — legs may be more usable underwater than from above, unlike the above-water expectation in the master blueprint |
| Partial bodies / frame edges | Joints outside the crop are reported as missing, never invented [Verified: [`ai/pose/base.py`](../src/lifeguard/ai/pose/base.py)] |
| Occlusion, bubbles, turbidity | Lower confidence, missed joints, left/right swaps (seen on back and inverted views [Verified, research doc §9]) |
| Camera angle | Head-down and inverted bodies were where models disagreed most [Verified, research doc §8] |
| Confidence | Only joints ≥ 0.5 are drawn [Verified: [`dashboard_controller.py`](../src/lifeguard/app/dashboard_controller.py)]; the pretrained model's confidence was the most informative of the four models tested (AUROC 0.77 on the AI reference) [Verified, research doc] |
| Stability over time | Jitter was measured in E0 (median second difference 0.033–0.053 of box size for YOLO) [Verified]; acceptability for any feature [TBD] |

Pose must not classify drowning, change the risk engine or raise alerts until it is separately validated and
explicitly approved [Verified current behaviour: the risk engine does not read `TrackObs.keypoints`;
[`POSE_INTEGRATION.md`](POSE_INTEGRATION.md)].

## G. Underwater camera calibration and geometry

### G.1 What exists [Verified: [`calibrated.py`](../src/lifeguard/system/geometry/calibrated.py)]

- Image → pool mapping is **one planar homography** estimated from ≥ 4 points, documented as "The water surface is a
  plane". A person counts as `in_water` when the **box centre**, projected onto that plane, lies inside the pool
  rectangle ([`real_engine.py`](../src/lifeguard/ai/perception/real_engine.py), `_to_pool`).
- Validation: point counts, points inside the image, non-singular homography, optional check points with a
  reprojection limit (default 0.3 m). Rectangular pools only.
- Zones: `EDGE` within `edge_band_m` of the border, `DEEP` for `x ≥ deep_from_x_m`, else `SHALLOW` — a **position along
  the pool**, not a measured water depth.
- No lens-distortion model and no refraction model in the geometry code.
- The calibration file format (`PoolCalibration`, one image size per file) can describe any single camera, so a second
  file for a second camera is possible; the runtime loads exactly one ([`system/config.py`](../src/lifeguard/system/config.py)).

### G.2 Why the underwater camera needs its own treatment

- **Position and orientation**: a different viewpoint entirely; the above-water calibration cannot be reused.
- **Wrong plane**: the existing model maps image points onto the water surface. A submerged swimmer is below it, so the
  projected position would be biased by depth [Assumption, to quantify]. A floor-plane homography is a candidate
  [Proposed], but a swimmer's box centre is neither on the floor nor on the surface.
- **Lens distortion**: wide-angle underwater housings commonly distort strongly [Assumption]; intrinsic calibration
  (e.g. a checkerboard filmed underwater) would be needed [Proposed].
- **Refraction**: light crosses water → housing window → air inside the housing; the effect depends on the window type
  (flat vs dome) and the angle. Whether a standard camera model calibrated underwater absorbs it adequately for our
  purpose is **not established here** [TBD]; no generic correction is claimed.
- **Pool floor reference**: lane lines and tiles on the floor are natural calibration points [Proposed].
- **Depth**: one camera gives no direct depth; anything like "near the bottom" needs an assumption (e.g. the body's
  image position relative to floor features) or extra sensors [TBD].
- **Relationship between the two coordinate systems**: both must map to the same pool frame (C-18: origin at corner 0,
  X along the length) for association (section D); the achievable agreement between the two mappings [TBD].

## H. Risk engine and safety logic

### H.1 Existing above-water logic — must stay unchanged [Verified]

[`risk/evidence.py`](../src/lifeguard/system/risk/evidence.py) and [`risk/engine.py`](../src/lifeguard/system/risk/engine.py):
evidence terms `submersion` (= seconds submerged / `t_sub_s`), `visibility` (share of the 4 s window in which the track
was **not** visible), `static` (low net speed in pool coordinates), `deep_zone`, `behavior` (weight 0); hard rule
submersion > `t_sub_s` → ALARM; detector distress → at most PRE_ALERT, off by default (`detector_distress_threshold:
null` in [`configs/system.yaml`](../configs/system.yaml)); FSM with hold times, hysteresis and cooldown; DISPATCH only
after a human CONFIRM.

### H.2 Why the same signals do not transfer to the underwater view

From above, "not visible" is evidence of being under water. From below, a submerged swimmer is typically **more**
visible, and a swimmer leaving the underwater camera's field of view may simply have surfaced or swum away. Feeding the
underwater camera into the current engine unchanged would invert the meaning of the `visibility` and `submersion`
terms. In addition, `in_water`, zones and the `static` term depend on the surface-plane geometry (section G). The
underwater camera must therefore **not** feed the operational risk engine as if it were another above-water camera.

### H.3 Candidate underwater evidence — research only [Proposed, not proven]

- A swimmer remaining motionless near the pool floor for a configurable duration [TBD] — the master blueprint notes
  that commercial systems (it names Poseidon, SwimEye, Lynxight, marked "verify citations before use") rely on a
  motionless-underwater timer (p. 33) [Verified that the blueprint says this; the commercial claim is not verified here].
- Absent or abnormal movement over a time window (e.g. vertical body, no propulsion).
- Pose/keypoint stability — **only** after pose is validated.
- Agreement or disagreement with the above-water view (e.g. surface track lost while the underwater view shows a still
  body at the same pool position).

None of these is a validated drowning indicator; no thresholds are proposed here.

### H.4 Behaviour under faults and conflicts [Proposed]

| Situation | Required behaviour |
| --- | --- |
| Underwater camera disconnects | Watchdog fault for that camera only; above-water detection, risk and alerts continue unchanged |
| Above-water camera disconnects | Existing above-water fault (INGEST/PERCEPTION, same priority as an alarm). Underwater observations must not silently replace the primary system; operator told that coverage is reduced |
| One camera delayed | Evidence older than a limit [TBD] is ignored for association/fusion; the primary pipeline never waits for the secondary |
| Conflicting evidence | Never suppress an above-water alert because of underwater evidence; conflicts are logged for review; what (if anything) underwater evidence may add is a future, approved decision |
| Swimmer visible in one camera only | Normal: each camera keeps its own tracks; no merging |
| Association uncertain | Treat as unassociated; do not merge or transfer risk history or incidents |

Human supervision remains mandatory: alerts are machine-generated and unconfirmed; dispatch needs a lifeguard CONFIRM
[Verified for the existing system].

## I. Image quality and environmental challenges [Assumption — to be measured on real footage]

| Challenge | Detection | Tracking | Pose |
| --- | --- | --- | --- |
| Bubbles and splash | Occluded or fragmented people; bubble clouds as false positives | Lost tracks, ID switches | Missing or wrong joints |
| Turbidity | Lower contrast, far swimmers missed | Shorter tracks | Lower confidence |
| Lighting from above (caustics, sun glare) | Moving bright patterns, false positives | Jittery boxes | Joint jitter |
| Surface reflections (mirror image seen from below) | Duplicate "person" above the waterline | Spurious tracks | Mirrored bodies (seen once in the pose data, frame 47155 [Verified: [`POSE_VITPOSE_INTEGRATION_ASSESSMENT.md`](POSE_VITPOSE_INTEGRATION_ASSESSMENT.md)]) |
| Camera housing (fouling, condensation, scratches) | Blur, haze | Degraded over days | Degraded |
| Lens distortion | Box shapes distorted at the edges | Position errors | Limb geometry distorted |
| Occlusion between swimmers | Merged boxes | ID switches | Joints assigned to the wrong person |
| Changing visibility (time of day, lights) | Recall varies | Variable | Variable |
| Compression and low-light noise | Artefacts, missed small swimmers | Box jitter | Joint jitter |

## J. Validation roadmap [Proposed]

| Stage | Objective | Required inputs | Work | Evidence | Exit criteria |
| --- | --- | --- | --- | --- | --- |
| 1. Camera and footage | Representative underwater recordings | The actual camera; pool access; safety supervision | Mount, record normal swimming and safe, supervised simulated scenarios (lifeguards present); log camera model, resolution, fps, mount position and angle | Recordings + metadata sheet | Coverage of scenarios agreed by the team [TBD] |
| 2. Existing-model inference | Baseline with no training | Stage 1 footage; Z5cc, ddt, YOLO11s-pose as registered | Offline runs; save per-frame results | Result files, overlay samples | Runs complete without errors |
| 3. Annotation and evaluation | Measure, not guess | Human-annotated evaluation subset (boxes, IDs, keypoints on a subset) | Precision/recall, missed swimmers, false detections, ByteTrack continuity, pose availability and agreement by body region | Evaluation report | Thresholds [TBD, approved by supervisor]; decide whether a gap is due to model, image quality, angle, configuration or calibration. Fine-tuning considered only here, only for a demonstrated gap |
| 4. Multi-camera runtime feasibility | Can both cameras run together? | Both cameras; target PC | Option 1 (two instances) first; measure FPS, GPU memory, latency; identify changes for option 2 | Measurements; design note | Above-water FPS and behaviour unchanged [TBD acceptance] |
| 5. Calibration | Underwater camera in pool coordinates | Calibration targets (floor lines, checkerboard) | Intrinsic + pool mapping; investigate cross-camera mapping | Calibration files, check-point errors | Error limits [TBD] |
| 6. Cross-camera association | Same swimmer in both views? | Synchronised recordings with reviewed identities | Test position/time-based association | Association accuracy, ambiguity cases | Accuracy [TBD]; otherwise remain unassociated |
| 7. Risk research | Underwater-specific evidence | Stages 3–6 | Candidate signals (H.3) computed offline | Signal behaviour on normal vs simulated scenarios | Signals worth shadow testing [TBD] |
| 8. Shadow-mode integration | Live logging without effect | Stages 4–7 | Both cameras live; underwater outputs logged only — no alerts, no drone actions | Logs vs human-reviewed scenarios | Agreed observation period [TBD] |
| 9. Controlled validation | Measured performance | Supervised sessions | Normal swimming + simulated incidents | False alarms, missed incidents, latency | Acceptance thresholds explicitly approved [TBD] |
| 10. Operational readiness | Go / no-go | All evidence | Document limitations | Decision record | Supervisor / project-team approval before any operational use |

## K. Safety, scope and non-goals

- The underwater camera is an **additional** camera, not a replacement; the above-water pipeline remains the primary
  system and its behaviour must remain unchanged.
- Existing trained models are reused first; additional training is not assumed and not currently required.
- Underwater support is **planned, not implemented or validated**.
- Pose remains display-only.
- No drowning diagnosis and no guaranteed detection is claimed.
- No autonomous physical response (including the simulated drone) may be triggered by unvalidated underwater
  predictions; the underwater instance runs with `response.platform: none`.
- Human supervision remains mandatory.

## L. Open questions and required decisions

| # | Question | Needs |
| --- | --- | --- |
| 1 | Which underwater camera (resolution, fps, lens/housing, interface: USB, RTSP, other)? | Camera specifications |
| 2 | Where and at what angle is it mounted; how much of the pool does it see? | Project team, pool owner |
| 3 | How do Z5cc / ddt / YOLO11s-pose perform on that camera? | Real underwater footage + annotation (J.2–J.3) |
| 4 | Two runtimes or one multi-camera runtime? | Measurements (J.4), architecture decision |
| 5 | One shared incident log or one per camera? One dashboard or two? | Architecture decision |
| 6 | Is the target PC (GTX 1650) enough for two pipelines with pose? | Performance evaluation |
| 7 | How is the underwater camera calibrated, and how accurate is the mapping to pool coordinates? | Calibration work (J.5) |
| 8 | Can swimmers be associated across views reliably enough to be useful? | Evaluation (J.6) |
| 9 | Which underwater signals, if any, may later contribute to risk, and with which thresholds? | Research (J.7–J.9), supervisor approval |
| 10 | Acceptance thresholds for detection, tracking, false alarms and latency | Supervisor approval |
| 11 | Privacy and consent for recording swimmers underwater | Pool owner, supervisor |
| 12 | Safety protocol for simulated incidents | Lifeguards, supervisor |

Related documents: [`SYSTEM_GUIDE.md`](SYSTEM_GUIDE.md) · [`ROADMAP.md`](ROADMAP.md) ·
[`MODEL_TRAINING_REPORT.md`](MODEL_TRAINING_REPORT.md) · [`POSE_INTEGRATION.md`](POSE_INTEGRATION.md) ·
[`POSE_ESTIMATION_BLUEPRINT.md`](POSE_ESTIMATION_BLUEPRINT.md) ·
[`POSE_VITPOSE_INTEGRATION_ASSESSMENT.md`](POSE_VITPOSE_INTEGRATION_ASSESSMENT.md).
