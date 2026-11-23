# Implementation Roadmap — Phase 2 onward

Status date: 2026-11-23. Scope: **swimming pools only**. Drone response is **simulation only**.

This roadmap is derived from the repository as it exists after Phase 1 and from the approved
planning documents (Engineering Blueprint M1–M13, `PHASE_1_*` documents, `MEMBER_1/2_NEXT_TASKS.md`,
`PHASE_1_DECISIONS_AND_ASSUMPTIONS.md`). Where documents disagree, the decision record wins
(e.g. D-02: the six Blueprint risk levels replace the four levels in `docs/SYSTEM_ARCHITECTURE.md`;
D-03: dispatch only on explicit human confirmation).

Labels used below: **REAL** = production-oriented implementation; **MOCK** = stand-in that replays
recorded data; **PLACEHOLDER** = interface or stub with no real behaviour; **DATA-BLOCKED** = cannot
be completed until the named dataset exists and is approved.

---

## 1. Verified starting point (Phase 1, complete)

| Area | Module | Kind | Notes |
| --- | --- | --- | --- |
| Data contract v1.0.0 (frozen) | `lifeguard.contracts` | REAL | `FrameResult`, `TrackObs`, `Submersion`, `Behavior`, `RiskEvent`, `Acknowledgement`, `DispatchCommand`, `SystemHealth`, `EngineHeartbeat`, `GroundTruthEvent`; ABCs `PerceptionEngine`, `PoolGeometry`, `RiskEngine`; JSONL/CSV I/O |
| Perception | `lifeguard.ai.perception.mock_engine` | MOCK | Replays fixture JSONL; no detection or tracking |
| Synthetic scenarios | `lifeguard.ai.sim` | REAL (test data generator) | 6 deterministic scenarios; they test logic, never model accuracy |
| Pool geometry | `lifeguard.system.geometry.stub` | PLACEHOLDER | Rectangular pool with affine pixel mapping; no calibration |
| Configs / logging | `lifeguard.ai.config`, `lifeguard.system.config`, `lifeguard.common` | REAL | YAML + pydantic, `extra="forbid"` |
| Integration | `lifeguard.app.replay_mock` | MOCK pipeline | Mock engine end to end; no risk logic |
| CI | `.github/workflows/ci.yml` | REAL | Windows + Ubuntu, Python 3.11 |

Not present in this repository: any pool-trained detector (the only weights,
`weights/seadrones_yolo11n_30ep_v1/best.pt`, are aerial/open-water and out of scope), tracking,
feature extraction, temporal classifier, risk engine, alerting, watchdog, drone code, real video ingest,
calibration.

### 1.1 Progress after Phase 1

| Batch | Items | Kind | Verified |
| --- | --- | --- | --- |
| 1 | P2-1 `lifeguard.system.risk` (RuleBasedRiskEngine) | REAL logic, starting thresholds | Yes: 52 risk tests, full suite 221 passed (user run, 2026-11-23) |
| 2 | P4-3 detector interface + `UltralyticsDetector`; P4-4 `ByteTracker` (NumPy); P4-5 `SubmersionMonitor` + `FeatureExtractor`; P4-6 `RealPerceptionEngine` + `engine: yolo` | REAL code; detector untested without the `vision` extra and weights | Yes: full suite 279 passed, ruff clean (user run, 2026-11-23) |
| 3 | P4-1 `VideoFileSource` (replay mode); `lifeguard.app.pipeline` + `run_video` CLI; P5 `BehaviorClassifier` interface + `RuleBehaviorBaseline` wired into the real engine (`behavior: rules`) | REAL ingest/CLI; behaviour baseline is an UNVALIDATED placeholder, risk weight stays 0 | Yes: 295 passed (user run); one Ruff finding fixed afterwards |
| 4 | P2-3 event evaluator `lifeguard.system.eval` + `evaluate` CLI with a one-parameter threshold sweep | REAL; on synthetic fixtures it is a logic check only | Yes: 310 passed, ruff clean (user run). Synthetic `distress_like` is MISSED at L_max 30 s (alarm only via submersion, ~50 s late): a real limitation of rules-only evidence |
| 5 | P5-1 sample builder `ai.temporal.dataset`; P5-2 trainable `SoftmaxBehaviorModel` (NumPy, JSON persistence), leave-one-session-out CV, `python -m lifeguard.ai.temporal.train`; `behavior: softmax` in `AIConfig` | REAL trainable code; NO real training data yet, so no trained model exists; risk weight stays 0 | Yes: 330 passed, ruff clean (user run) |
| 6 | P3-1/P3-2 `lifeguard.system.alerts`: `AlertService` (incidents, PRE_ALERT/ALARM notifications, unacknowledged-ALARM repeats that never dispatch, `DispatchCommand` only from a lifeguard CONFIRM), SQLite `IncidentStore`, `alerts:` block in `system.yaml` | REAL logic; delivery channels (dashboard, siren, push) not built; not yet wired into `run_video` (Phase 7 runtime) | Yes: 341 passed (user run); one Ruff finding in a test fixed afterwards |
| 7 | P6-1 `lifeguard.robotics.mission`: `MissionPlanner` (stand-off hover point clear of all people, geofence, altitude ceiling, pre-flight battery/link checks, pool -> NED waypoints) and `MissionManager` (one drone, abort with RTL on dismissal/close, follows `merged_from`); `mission:` block in `system.yaml` | REAL planning logic, SIMULATION ONLY | Yes: 355 passed (user run) |
| 8 | P3-4 `system.health.Watchdog` (camera loss, frozen video, low FPS, perception errors, disk space) with SYSTEM FAULT notifications; P3-3 `system.recording.ClipRecorder` (pre/post alarm clips, linked in the incident log) | REAL | Yes (agent run, see section 1.2) |
| 9 | Phase 7 `app.runtime.LifeguardRuntime` + `python -m lifeguard.app.main` (recorded video, live camera, synthetic demo); `system.ingest.live.LiveCameraSource` (latest-frame policy, reconnect); `system.response` platform interface + `ManualResponse` | REAL; live camera tested with a fake capture only | Yes (agent run) |
| 10 | P4-2 `system.geometry.calibrated` (normalised-DLT homography, validation, JSON) + `python -m lifeguard.system.geometry.calibrate`; `geometry:` block selects calibration or a labelled PLACEHOLDER | REAL; a real calibration needs a pool camera frame | Yes (agent run) |
| 11 | P3-5 Streamlit dashboard `app.dashboard` + headless `app.dashboard_controller` (live view, tracks/risk, incidents with Confirm/Dismiss, response + release confirmation, health, history) | REAL UI (`dashboard` extra) | Yes (agent run) |
| 12 | P6-2/P6-3 `robotics.executors`: `SimulatedExecutor` (pure Python, failsafes), `MavlinkExecutor` (ArduPilot SITL, loopback endpoints only, human-confirmed simulated release), `robotics.response.DroneResponse` | SIMULATION ONLY; MAVLink executor tested against a scripted fake connection, NOT against SITL | Yes (agent run, fake connection) |
| 13 | Remaining software-only tasks from the official task lists: M2-12 model registry (`common.models`, `models/registry.yaml`, SYSTEM FAULT on integrity failure); M1-03 `feature_spec.json` + `bob_freq_hz`; M1-04 annotation guide, label taxonomy, interval CSV, annotator agreement, interval-based training; checklist N-consecutive-errors SYSTEM FAULT; MOT/TrackEval export (M1-05 tooling); milestone tests I0 and I3 (pure-Python simulation); CI dashboard job; `docs/index.md`; CODEOWNERS template | REAL software; no trained models, datasets or hardware involved | Yes (agent run, location `E:\lifeguardDron\lifeguard-system`) |

### 1.2 Verification recorded by the agent (2026-11-23)

From batch 8 on, the agent ran the tests itself (authorised by the user). Final results are in the handoff
report and reproduced by `python -m ruff check src tests` and `python -m pytest`. See
[`SYSTEM_GUIDE.md`](SYSTEM_GUIDE.md) for the A/B/C verification levels: software tested (A) — yes;
hardware/SITL (B) — no; AI validated on real pool recordings (C) — no.

Batch 2 notes: the tracker uses greedy IoU assignment (no SciPy) and freezes lost tracks in place; the
submersion re-association radius starts at 2.0 m (Blueprint: 1.5 m) to allow one frame of tracker
confirmation delay. The pipeline is verified only on synthetic boxes; detection quality on pool video is
unmeasured. Observed in batch 5 testing: in the synthetic `distress_like` scenario (head bobbing plus short
gaps) the tracker gives the subject new IDs after some gaps (IDs 1 -> 4 -> 5); the submersion monitor links
them through `merged_from` and labels and risk history follow the chain. ID fragmentation should be
measured (IDF1/ID switches) and the lost-track matching threshold tuned on annotated pool clips (P4-4).

---

## 2. Architecture target

```
video ingest ─► PerceptionEngine ─► FrameResult ─► RiskEngine ─► RiskEvent ─► AlertService ─► operator
 (M1)           (M3 detect, M5 track,              (M8 score +                (M9 incidents,      │ ack
                 M6 features + submersion,          FSM)                       re-alert, clips)   ▼
                 M7 behaviour)                         ▲ Acknowledgement ◄──────────────────────── │
                                                                    DispatchCommand (human only) ──┘
                                                                          ▼
                                                    MissionPlanner (M10) ─► MAVLink ─► ArduPilot SITL
Watchdog (M13) ◄── SystemHealth / EngineHeartbeat from every component
```

Package layout (extends decision C-01; no top-level package changes):

| Package | Owner | Phase |
| --- | --- | --- |
| `lifeguard.system.risk` | Member 2 | 2 |
| `lifeguard.system.eval` | Member 2 | 2 |
| `lifeguard.system.alerts`, `lifeguard.system.health` | Member 2 | 3 |
| `lifeguard.system.ingest`, `lifeguard.system.geometry.calibrated` | Member 2 | 4 |
| `lifeguard.ai.detection`, `lifeguard.ai.tracking`, `lifeguard.ai.features`, `lifeguard.ai.perception.real_engine` | Member 1 | 4 |
| `lifeguard.ai.temporal` | Member 1 | 5 |
| `lifeguard.robotics.mission`, `lifeguard.robotics.mavlink` | Member 2 | 6 |
| `lifeguard.app.main` | Member 2 | 7 |

Heavy dependencies go into optional extras so the core install and CI stay small:
`vision` (opencv-python, ultralytics), `temporal` (torch, scikit-learn), `dashboard` (streamlit),
`drone` (pymavlink). Tests that need an extra use `pytest.importorskip` and run in a separate CI job.

---

## 3. Phases

### Phase 2 — Decision core (no data dependency)

| ID | Deliverable | Kind | Depends on | Acceptance criteria | Tests |
| --- | --- | --- | --- | --- | --- |
| P2-1 | `RuleBasedRiskEngine` (Blueprint M8, MEMBER_2 M2-05): evidence terms, EMA risk, 6-level FSM with hysteresis, dwell, hard submersion rule, cooldown, `merged_from` history transfer, acknowledgements | REAL (rules); thresholds are starting values | contracts | A unit test per transition, hard rule from every state; 0 flicker on a sine-wave risk crossing a threshold every 0.5 s; every synthetic `submersion` event reaches ALARM; no `dive`/`edge_exit`/`normal_swim`/`idle` reaches ALARM; all thresholds in `configs/system.yaml` | `tests/system/risk/` |
| P2-2 | Risk replay CLI `lifeguard.app.replay_risk`: fixture → engine → `RiskEvent` JSONL + summary | REAL CLI over MOCK perception | P2-1 | Exit codes as SPEC §8; deterministic output on repeated runs | subprocess tests |
| P2-3 | Event-level evaluator `lifeguard.system.eval` (M2-06): matched/missed events, FA/h, detection latency, threshold sweep | REAL | P2-2 | Hand-computed cases pass (e.g. 3 events, 2 matched, 1 FA in 0.5 h → FA/h = 2.0); matching rule documented: alarm for the correct track between onset and onset + `L_max` | unit tests |

### Phase 3 — Alerting, operator loop and health

| ID | Deliverable | Kind | Depends on | Acceptance criteria | Tests |
| --- | --- | --- | --- | --- | --- |
| P3-1 | `AlertService`: incident store (SQLite, stdlib), priority mapping (PRE_ALERT visual, ALARM visual+sound), ALARM re-alert every `T_ack` while unacknowledged (no level change, no dispatch — D-03), dismissals stored as hard negatives | REAL | P2-1 | Synthetic submersion → incident row; dismissal row has timestamp; unacknowledged ALARM re-alerts and never dispatches | unit + replay tests |
| P3-2 | `DispatchCommand` creation **only** from a CONFIRM acknowledgement on an ALARM, with `confirmed_by = Acknowledgement.by` | REAL | P3-1 | No code path creates a `DispatchCommand` without a human ack (test enumerates paths) | unit tests |
| P3-3 | Clip ring buffer (last 10 s in memory; 10 s before + 5 s after an ALARM written to `events/`) | REAL | P4-1 | Clip duration within ±1 frame of target | uses synthetic frames |
| P3-4 | Health watchdog (M13): heartbeats, frozen-frame hash, FPS floor, disk space; SYSTEM_FAULT as `SystemHealth(status=error)` | REAL | contracts | Killed engine / frozen video / missing camera → fault within 3 s | fault-injection tests |
| P3-5 | Operator dashboard (Streamlit, `dashboard` extra): live overlay, track risk bars, alarm banner, confirm/dismiss | REAL UI | P3-1 | Synthetic submersion shows ALARM, confirm produces DISPATCH + `DispatchCommand` | manual checklist + smoke test |

### Phase 4 — Real perception pipeline

| ID | Deliverable | Kind | Depends on | Acceptance criteria | Tests |
| --- | --- | --- | --- | --- | --- |
| P4-1 | Video ingest (M2-03): file replay (deterministic) and live/RTSP latest-frame reader with reconnect | REAL | — | Replay gives identical `frame_idx` sequences; live mode drops stale frames and logs the count; reconnect < 5 s | synthetic video written in the test |
| P4-2 | Calibrated `PoolGeometry` (M2-04): homography from ≥4 clicked corners, polygons, exits, pool→NED transform in `configs/pool_<site>.json` | REAL | pool calibration frame | Reprojection error < 0.3 m on known marks | unit tests on hand-picked points |
| P4-3 | Detector wrapper `Detector.detect(frame)` on Ultralytics YOLO11n, COCO-pretrained `person` first | REAL (pretrained) | `vision` extra | Runs on GTX 1650 at batch 1; failure → `FrameResult(status=error)` | importorskip tests + recorded-video smoke test |
| P4-4 | Tracker (ByteTrack two-stage association, track buffer, `merged_from` on re-association within 1.5 m) | REAL | P4-3 | IDF1/HOTA measured on 3–5 annotated pool clips (DATA-BLOCKED for the metric, not for the code) | unit tests on synthetic detections |
| P4-5 | Per-track features + submersion detector (M1-03/M6) filling `TrackObs.features` and `FrameResult.submersions` | REAL (rules) | P4-2, P4-4 | Reproduces the synthetic disappearance model exactly when fed synthetic detections | unit tests |
| P4-6 | `RealPerceptionEngine` + factory `engine: yolo` (minor `AIConfig` extension) | REAL | P4-3..5 | Same `FrameResult` contract; `process()` never raises; heartbeat with fps/latency | contract tests reused |
| P4-7 | Pool-specific detector fine-tune (M1-10) | REAL, **DATA-BLOCKED** | approved pool person/head dataset | Recall target on in-water people (target set by the team) on a source-split test set | evaluation report |

### Phase 5 — Temporal behaviour classifier (M7)

| ID | Deliverable | Kind | Depends on | Acceptance criteria |
| --- | --- | --- | --- | --- |
| P5-1 | Window builder over P4-5 features | REAL | P4-5 | Deterministic windows from fixtures |
| P5-2 | Baselines: rules-only → gradient-boosted trees → GRU/TCN producing `Behavior` | REAL, **DATA-BLOCKED** | staged pool sessions (P1b) | Leave-one-session-out results; beats rules-only on event recall |
| P5-3 | Enable the risk engine's behaviour weight (currently 0) | config change | P5-2, P2-3 | Evaluator shows lower miss rate at equal or lower FA/h |

### Phase 6 — Drone response (simulation only)

| ID | Deliverable | Kind | Depends on | Acceptance criteria |
| --- | --- | --- | --- | --- |
| P6-1 | `MissionPlanner` (pure Python): `DispatchCommand` → pool→NED target, hover point offset 1.5 m from the swimmer, inside the geofence, never over the swimmer zone; abort on dismissal | REAL | P3-2, `PoolGeometry.to_local_ned` | Unit tests for offset, fence rejection, abort |
| P6-2 | MAVLink adapter (pymavlink, `drone` extra): GUIDED, arm, takeoff, goto, servo release (`MAV_CMD_DO_SET_SERVO`), RTL | REAL against SITL | ArduPilot SITL (WSL2/Ubuntu) | Scripted mission 10/10 runs; failsafes (battery, link loss, cancel) end in RTL/LAND |
| P6-3 | Simulated payload release | SIMULATED | P6-2 | Servo PWM change acknowledged in 10/10 runs |

No code path flies real hardware. Release is never triggered by the AI alone.

### Phase 7 — Integrated runtime (`lifeguard.app.main`)

Single process, bounded queues, latest-frame policy, config-selected engine (mock → pretrained → fine-tuned),
watchdog, alert service, optional mission planner. Acceptance: end-to-end replay of a synthetic and a recorded
pool video produces the expected incidents; latency per stage logged; CI runs the mock configuration.

### Phase 8 — Evaluation and tuning (M2-14, Blueprint P11)

Fit evidence weights and thresholds on held-out real sessions (**DATA-BLOCKED**), report miss rate vs FA/h,
detection latency, IDF1/HOTA, end-to-end latency on the target GPU, with stated limitations.

---

## 4. Dependency order

```
P2-1 ─► P2-2 ─► P2-3 ───────────────────────────────┐
  └──► P3-1 ─► P3-2 ─► P6-1 ─► P6-2 ─► P6-3          │
         └──► P3-5                                   ├─► P7 ─► P8
P3-4                                                 │
P4-1 ─► P3-3                                         │
P4-3 ─► P4-4 ─► P4-5 ─► P4-6 ───────────────────────┘
P4-2 ─────────────┘        └─► P5-1 ─► P5-2 (data) ─► P5-3
```

Can start now without data: P2-*, P3-1/2/4, P4-1..P4-6 (pretrained COCO), P6-1.
Blocked on data or external setup: P4-7, P5-2, P6-2/3 (SITL environment), P8.

---

## 5. Contract policy

Contract v1.0.0 stays frozen. Planned additions are **minor** versions owned as in C-12:
`FeatureSpec` (P4-5), `PoolGeometryConfig` (P4-2), `MissionStatus` (P6-1/2). Known constraint:
`RiskEvent` requires `to_level != from_level`, so an un-acknowledged ALARM's periodic re-alert is an
`AlertService` action, not a `RiskEvent` (no contract change needed).

---

## 6. Open decisions (need the team)

| ID | Question | Current handling |
| --- | --- | --- |
| OD-1 | Should an un-acknowledged PRE_ALERT escalate to ALARM after a timeout (`ACK_TIMEOUT_ESCALATION`)? | Implemented as `risk.pre_alert_ack_timeout_s`, **disabled** (`null`) by default |
| OD-2 | Risk weights and thresholds | Blueprint starting values; must be tuned in P8 before any real use |
| OD-3 | Copy the six `docs/phase1/` documents into the repo | Pending approval |
| OD-4 | CODEOWNERS usernames (H-04) | Not created |
| OD-5 | Branch for Phase 2 work (`main` vs `int/phase1-foundation`) | Work is uncommitted |
| OD-6 | Latency budget and acceptable FA/h (O-04) | 100 ms placeholder in `system.yaml` |
| OD-7 | Ultralytics (and its built-in trackers) is AGPL-3.0; publishing implications | Decide before P4-3 is merged |
| OD-8 | Which pool dataset is approved for P4-7 and which staged-session protocol for P5-2 | See `dataset_recovery/` reports |
| OD-9 | `docs/SYSTEM_ARCHITECTURE.md` still shows the superseded 4-level FSM (O-06) | Left unchanged; this roadmap follows D-02 |
