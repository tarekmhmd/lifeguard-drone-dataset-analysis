# System Guide — Autonomous AI Lifeguard System (pool only)

Status date: 2026-11-25. This guide describes the code as it exists in this repository. Planning history is
in [`ROADMAP.md`](ROADMAP.md); the pool detectors and how they were trained are in
[`MODEL_TRAINING_REPORT.md`](MODEL_TRAINING_REPORT.md); the supervisor demo is in [`DEMO.md`](DEMO.md); the frozen data contract is `src/lifeguard/contracts` (v1.0.0).

> Decision support only. The system raises *suspected* incidents for lifeguards to check; it never
> declares a drowning and never acts physically on its own. Drone response is simulation only.

## 1. Verification status

| Level | Meaning | Status |
| --- | --- | --- |
| **A. Software complete and tested** | Every component implemented, unit/integration tests pass on CPU without weights, cameras or SITL | **Yes**: `python -m pytest` (see the ROADMAP progress table for the latest recorded run) |
| **B. Hardware / simulator integration verified** | Real camera, ArduPilot SITL, real YOLO weights exercised end to end | **Partly.** The real engine runs end to end with the trained pool weights on a GTX 1650 (about 14 ms/frame, recorded video and dashboard). Live-camera code still tested only with a fake capture; MAVLink executor only against a scripted fake connection |
| **C. AI validated on real pool recordings** | Detection, tracking, behaviour and alarm performance measured on labelled pool video | **Partly.** Both detectors are evaluated on held-out labelled internet pool videos (Z5cc test mAP50 0.704, ddt 0.636), and the early-warning rule on 12 test clips. Not yet on the team's own pool camera; tracking and false alarms per hour unmeasured |

## 2. Architecture and data flow

```
 VideoFileSource / LiveCameraSource / SyntheticFrameSource          (lifeguard.system.ingest)
        │ VideoFrame (image, frame_idx, t_capture, t_utc, t_media)
        ▼
 PerceptionEngine ── Mock (fixture replay) | Real:                   (lifeguard.ai)
        │            Detector (YOLO, optional) → ByteTracker → pool mapping (PoolGeometry)
        │            → SubmersionMonitor + FeatureExtractor → BehaviorClassifier (optional)
        │ FrameResult (contract)
        ▼
 RuleBasedRiskEngine  (evidence → EMA risk → 6-level FSM, hard submersion rule)   (lifeguard.system.risk)
        │ RiskEvent (contract)
        ▼
 AlertService (incidents, notifications, repeats, SQLite log, clips)   (lifeguard.system.alerts)
        │ ▲ Acknowledgement (lifeguard, via dashboard)
        │ DispatchCommand — only from a human CONFIRM on an ALARM
        ▼
 ResponsePlatform: ManualResponse (fixed camera / alert only)          (lifeguard.system.response)
                 | DroneResponse (SIMULATION): MissionManager → SimulatedExecutor | MavlinkExecutor(SITL)
 Watchdog (health faults) · ClipRecorder (alarm clips)                 (lifeguard.system.health / recording)
 LifeguardRuntime wires all of the above; main.py (CLI) and dashboard.py (Streamlit) drive it. (lifeguard.app)
```

**AI core independence.** `lifeguard.ai`, `lifeguard.system.risk` and `lifeguard.system.alerts` never import
`lifeguard.robotics`. The response platform is chosen by `response.platform` in `configs/system.yaml`
(`none` by default, because the supervisor has not chosen drone vs fixed camera vs hybrid).

**Extension points**
- New response (e.g. poolside siren zone, rescue-tube dispenser, pager): implement `ResponsePlatform`
  (`lifeguard.system.response`) and select it in `build_response` (`lifeguard.app.runtime`).
- New detector: implement `Detector` (`lifeguard.ai.detection.base`).
- New behaviour model: implement `BehaviorClassifier` (`lifeguard.ai.temporal.base`); `SoftmaxBehaviorModel`
  is the trainable reference.
- New geometry (non-rectangular pool): implement `PoolGeometry` (contract interface).
- New vehicle interface: implement `MissionExecutor` (`lifeguard.robotics.executors.base`).

## 3. Components

| Component | Module | Kind | Notes |
| --- | --- | --- | --- |
| Data contract v1.0.0 | `lifeguard.contracts` | REAL (frozen) | unchanged since Phase 1 |
| Mock perception | `ai.perception.mock_engine` | MOCK | replays synthetic fixtures |
| Real perception | `ai.perception.real_engine` | REAL pipeline | accuracy unmeasured on pool video |
| YOLO detector | `ai.detection.ultralytics_detector` | REAL, optional (`vision` extra) | pool weights: Z5cc side-view, ddt overhead (`models/`, not committed); several classes → PERSON with class-agnostic NMS; reports the distress-class confidence; AGPL-3.0 (OD-7) |
| Tracker | `ai.tracking.bytetrack` | REAL (NumPy) | greedy matching; lost tracks frozen in place |
| Features + submersion timers | `ai.features` | REAL (rules) | 2.0 m re-association radius; the real engine adds `det_distress_2s` (2 s mean of the detector's drowning confidence; not part of the feature spec) |
| Behaviour: rules baseline | `ai.temporal.rules` | PLACEHOLDER baseline, unvalidated | off by default |
| Behaviour: softmax model | `ai.temporal.softmax`, `train`, `cv` | REAL trainable code | no trained model exists (no labelled data) |
| Risk engine | `system.risk` | REAL | thresholds are starting values; behaviour weight 0; optional detector-distress rule → PRE_ALERT (`DETECTOR_DISTRESS`, never ALARM), off unless `detector_distress_threshold` is set |
| Event evaluator | `system.eval`, `app.evaluate` | REAL | synthetic results are logic checks only |
| Alerts + incident log | `system.alerts` | REAL | channels (siren, push) not built; dashboard shows them |
| Health watchdog | `system.health` | REAL | camera loss, frozen video, low FPS, perception errors, disk |
| Alarm clips | `system.recording` | REAL | needs OpenCV and `recording.out_dir` |
| Calibration | `system.geometry.calibrated`, `calibrate` | REAL | needs a real camera frame to produce a real calibration |
| Placeholder geometry | `system.geometry.stub` | PLACEHOLDER | used only without a calibration, and labelled as such |
| Response: manual | `system.response.ManualResponse` | REAL | fixed-camera / alert-only deployments |
| Mission planner | `robotics.mission` | REAL planning, SIMULATION use | stand-off 1.5 m, geofence, abort on dismissal |
| Simulated executor | `robotics.executors.simulated` | SIMULATION | kinematic model, no physics |
| MAVLink executor | `robotics.executors.mavlink` | SIMULATION (SITL only) | loopback endpoints only; never run against SITL here |
| Runtime + CLI | `app.runtime`, `app.main` | REAL | recorded video, live camera, synthetic demo |
| Dashboard | `app.dashboard`, `app.dashboard_controller` | REAL UI (`dashboard` extra) | buttons call the runtime |
| Model registry (M2-12) | `common.models`, `models/registry.yaml` | REAL | sha256 per model; unregistered/missing/mismatched file → SYSTEM FAULT, no start; only MOCK fixtures registered so far |
| Feature spec (M1-03) | `ai.features.spec`, `ai/features/feature_spec.json` | REAL | 8 features, 4 s window, 0.5 s step; drives the model input vector |
| Behaviour labels (M1-04) | `ai.temporal.labels`, `label_taxonomy.json`, `docs/ai/annotation_guide.md` | REAL | interval CSV, validation, annotator agreement (kappa), interval-based training |
| MOT export (M1-05 tooling) | `ai.tracking.mot` | REAL | TrackEval layout; metrics need annotated clips |
| Milestone tests | `tests/integration/` | REAL tests | I0 (mock end to end), I3 (pure-Python drone simulation, not SITL) |

## 4. Safety behaviour (implemented and tested)

- A detection is never a drowning: alarms come only from the risk engine's temporal evidence and the hard
  submersion rule (submerged > `t_sub_s`, default 10 s). No single-frame rule exists.
- The behaviour classifier's weight in the risk score is 0 until a model is validated on real sessions.
- The detector's drowning class never raises an ALARM. Its 2 s mean can raise a PRE_ALERT ("check this
  swimmer") only when `risk.detector_distress_threshold` is set; it is `null` in `configs/system.yaml`.
- ALARM and DISPATCH are never cleared automatically; an unacknowledged ALARM is repeated every 15 s and
  never dispatches (decision D-03).
- A `DispatchCommand` is created only from a lifeguard CONFIRM on an ALARM (`AlertService.acknowledge`).
- All messages say "machine-generated alert, unconfirmed"; none claims a drowning was detected.
- Health faults are raised with alarm priority ("watch the pool directly"). Perception faults after
  `max_consecutive_engine_errors` (10) error frames in a row; fewer are reported as DEGRADED.
- A model file that is not registered, missing, or fails its sha256 check stops start-up with a SYSTEM FAULT.
- Drone: hover 1.5 m from the swimmer and from everyone else, inside the geofence, battery/link checks,
  one mission at a time, abort with RTL on dismissal; release waits for a human confirmation
  (`require_release_confirmation: true`); the MAVLink executor refuses any non-loopback endpoint.

## 5. Configuration (`configs/system.yaml`, `configs/ai*.yaml`)

| Block | Purpose | Key defaults |
| --- | --- | --- |
| `risk` | evidence weights, thresholds, dwell, cooldown | `t_sub_s 10`, behaviour weight 0, `pre_alert_ack_timeout_s null` (OD-1) |
| `alerts` | repeats, incident database | `realert_interval_s 15`, `incident_db null` (memory) |
| `health` | watchdog thresholds | camera/frozen 3 s, FPS floor 5, 1 GB free disk |
| `recording` | alarm clips | `out_dir null` (disabled), 10 s before / 5 s after |
| `geometry` | calibration file or placeholder | `calibration null` → PLACEHOLDER rectangle |
| `mission` | planner | home (-1, -1), stand-off 1.5 m, fence margin 2 m, ceiling 10 m, battery 40 % |
| `response` | platform + drone executor | `platform none`; drone `executor simulated`, release needs confirmation |
| `ai.yaml` | MOCK engine | submersion fixture, `model_registry: ../models/registry.yaml` |
| `ai_yolo.yaml` | REAL engine, COCO YOLO11n | local weights path (must be registered), `behavior: none` |
| `ai_z5cc.yaml` | REAL engine, **side-view pool detector** | `models/z5cc_s448_final.pt` (registered), imgsz 448, classes [0,1] → PERSON, distress class 1 |
| `ai_ddt.yaml` | REAL engine, **overhead pool detector** | `models/ddt_v2_final.pt` (registered), imgsz 640, classes [0,1,2] → PERSON, distress class 1 |
| `system_demo_drone.yaml` | DEMO copy of `system.yaml` | `response.platform: drone` (simulation), `detector_distress_threshold: 0.5` |
| top level | engine error limit | `max_consecutive_engine_errors: 10` (watchdog SYSTEM FAULT threshold) |

## 6. Optional dependencies

```powershell
python -m pip install -e ".[dev]"         # tests and lint (required)
python -m pip install -e ".[vision]"      # Ultralytics YOLO + OpenCV (real detector, video, clips)
python -m pip install -e ".[dashboard]"   # Streamlit operator dashboard
python -m pip install -e ".[drone]"       # pymavlink for ArduPilot SITL
```

Missing optional parts fail safely: no Ultralytics/weights → exit code 3 (`ModelLoadError`); no SITL →
the runtime falls back to manual response and reports `UNAVAILABLE drone`; no clip directory → recording
disabled; no calibration → PLACEHOLDER geometry, reported at startup.

## 7. Commands

```powershell
# Full stack without camera or weights (MOCK perception, real everything else)
python -m lifeguard.app.main --ai-config configs/ai.yaml --system-config configs/system.yaml --synthetic

# Operator dashboard (then choose an input and press Start)
streamlit run src/lifeguard/app/dashboard.py

# Recorded video / live camera with the real engine (vision extra + local weights)
python -m lifeguard.app.main --ai-config configs/ai_yolo.yaml --system-config configs/system.yaml --video pool.mp4
python -m lifeguard.app.main --ai-config configs/ai_yolo.yaml --system-config configs/system.yaml --camera 0

# Calibrate a camera (corners from any image viewer), then set geometry.calibration in system.yaml
python -m lifeguard.system.geometry.calibrate --site-id my_pool --length 25 --width 12.5 --image frame.png --corners "212,180;1705,175;1880,950;40,960" --exit 0,2 --out configs/pool_my_pool.json

# Evaluate the risk engine against labels; train a behaviour model from real-engine frames
python -m lifeguard.app.evaluate --frames frames.jsonl --truth events.csv --system-config configs/system.yaml
python -m lifeguard.ai.temporal.train --frames s1.jsonl --truth s1.csv --frames s2.jsonl --truth s2.csv --cv --out models/behavior.json
python -m lifeguard.ai.temporal.train --intervals --frames s1.jsonl --truth s1_labels.csv --frames s2.jsonl --truth s2_labels.csv --cv --out models/behavior.json

# Labels: validate a file, measure agreement between two annotators
python -m lifeguard.ai.temporal.labels validate s1_labels.csv
python -m lifeguard.ai.temporal.labels agreement annotator_a.csv annotator_b.csv

# Model registry: verify every entry, checksum a new file, fetch a registered URL (explicit only)
python -m lifeguard.common.models verify
python -m lifeguard.common.models hash models/yolo11n.pt

# Tracking evaluation input (TrackEval MOTChallenge layout) from real-engine frames
python -m lifeguard.ai.tracking.mot --frames run_frames.jsonl --root eval --seq pool_clip01 --fps 10
```

Drone in simulation: set `response.platform: drone` (pure-Python simulator by default). For ArduPilot SITL
set `response.drone.executor: mavlink_sitl`, install the `drone` extra, start SITL yourself
(e.g. `sim_vehicle.py -v ArduCopter --out=udp:127.0.0.1:14550` in WSL2/Ubuntu) and make sure the
calibration's pool → NED transform matches the vehicle's EKF origin.

## 8. What still needs real data, hardware or decisions

- **Real data:** the pool detectors exist (P4-7 done); still needed: recordings from the team's own pool
  camera to measure them in deployment, labelled pool sessions to train and validate the behaviour model
  and to tune risk thresholds (P5-2, P8), and annotated clips for tracking ID-switch tuning.
- **Hardware / simulator:** a pool camera (live mode, calibration frame); ArduPilot SITL for the MAVLink
  executor; no physical drone or release mechanism is supported.
- **Supervisor / team decisions:** response platform (drone, fixed camera, hybrid); OD-1 PRE_ALERT
  ack-timeout escalation; OD-6 acceptable false alarms per hour and latency budget; OD-7 Ultralytics
  AGPL-3.0; OD-3 copying `docs/phase1/` into the repo; OD-4 CODEOWNERS; OD-5 branch.
