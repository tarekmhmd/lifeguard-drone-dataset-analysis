# Multi-source data strategy — research proposal and critical assessment

Date: 2026-11-27. Discussion document only: nothing downloaded, trained or changed.
Legend: **[V]** verified (repository documents, or source pages read earlier: see `POSE_SWIMXYZ_STRATEGY.md`) ·
**[A]** assumption or estimate · **[R]** recommendation.

## 1. Verdict

The three-source idea is **scientifically sound in structure**: it gives each task the data that can actually teach
it, instead of hoping one dataset does everything. It needs four corrections to be defensible:

1. **Source C cannot measure real drowning detection.** Acted distress is voluntary and safe, and it follows what
   the actor thinks distress looks like. Real distress (the instinctive drowning response) is involuntary,
   often silent and short. C can test whether the system **responds to the observable indicators we defined**.
   It cannot estimate the real-world miss rate. Every result on C must be reported as "on simulated distress".
2. **Source B is the most valuable and the easiest to evaluate honestly.** False alarms per hour on long normal
   sessions decide whether lifeguards would accept the system, and this can be measured for real. B should get
   most of the collection effort, not C.
3. **Camera placement must be fixed before filming.** Pose and behaviour models learn the camera. Footage
   from a phone at the pool edge will not represent an underwater, overhead or drone camera. Decide the target
   view(s) first, then collect only with those views.
4. **One pool is not enough for generalisation claims.** Collect from at least two pools or clearly different
   conditions (lighting, time of day, crowd level), and keep one of them for testing only.

A fourth, existing signal deserves more weight than new learned models: the **submersion timer** (hard rule:
submersion > 10 s raises ALARM [V]). Source B footage of normal underwater swimming and diving is what calibrates
it against false alarms.

## 2. Three tasks, kept separate

| Task | Question | Learns from | Output in our pipeline |
| --- | --- | --- | --- |
| 1. Pose estimation | where are the joints? | A (synthetic) + real pose annotations from B and C frames | `TrackObs.keypoints` (17 COCO points, contract v1.0.0 [V]) |
| 2. Behaviour analysis | what is this person doing over time? | B (normal activities) + C (simulated indicators) | per-track activity and indicator evidence (features) |
| 3. Risk assessment | does the evidence justify an alert? | B (false alarms), C (response to indicators), existing rules | the existing risk FSM + alerts [V] |

Rules that keep them apart [R]: pose is evaluated only against keypoint ground truth; behaviour only against
interval labels; risk only at event level (alerts per session). An unusual pose is an observation, not a label.

## 3. What each source contributes

| Source | Contributes | Does not contribute |
| --- | --- | --- |
| **A. SwimXYZ** (synthetic, 4 strokes, 5 views, COCO17 2D/3D joints, CC BY 4.0 [V]) | joint localisation across strokes and views; clean normal-stroke keypoint sequences | real water and people; treading water, floating, standing; any distress; multi-swimmer scenes |
| **B. Real normal footage** | real domain for pose; the "normal" distribution; false-alarm measurement; submersion-timer calibration | positive (distress) examples |
| **C. Simulated distress** | real-domain examples of the indicators we defined; test cases for time-to-alert; transitions | real drowning dynamics; real miss rates |
| *Existing:* Z5cc, ddt, `datasets (1)` | detector data; a few internet near-drowning clips [V] | provenance of the drowning labels in `datasets (1)` is unknown [V]; no video ids; must not be mixed into C |

## 4. Remaining gaps

- No real drowning footage (by design; not assumed available).
- Unconscious or passive submersion cannot be simulated safely. Passive submersion is detected only through
  the submersion timer, which must stay the primary rule.
- Children and weak swimmers: the highest-risk group, but the hardest to film ethically. Results on adult actors
  do not transfer automatically [A].
- Crowded public sessions: closed filming sessions will have fewer people than real pool hours.
- Night and indoor lighting variation, if the pool operates in both.

## 5. Dataset structure [R]

One record per **session** (one filming day at one pool with fixed cameras), never split across train and test.

```
sessions/<session_id>/
  session.json        pool_id, date, cameras (id, view: underwater/overhead/edge/drone, calibration ref),
                      consent_record_ref, source (B|C), protocol_version
  videos/<camera_id>.mp4            raw, restricted access
  tracks/<camera_id>.json           detector boxes + ByteTrack ids (generated, versioned by model sha256)
  pose/<camera_id>.coco.json        sparse frames, COCO keypoints 1.0, per-frame provenance
  events/<camera_id>.csv            interval labels (schema below), one row per interval per reviewer
  script/<run_id>.json              C only: scripted phase times from the protocol log
participants.csv (separate, restricted)   participant_id, role (actor/swimmer), consent scope; no names in data
```

Splits [R]: by **participant and session** for behaviour and risk; by session for pose; one pool held out as
the final test where possible.

## 6. Annotation strategy

**Use two schemas, not one [R].**

1. **Pose: COCO keypoints** (17 points, matches the system contract [V]). Sparse sampled frames only (for example,
   1 frame every 2-5 s per track). Corrected from model proposals in CVAT, with `provenance`
   (`human` / `model_proposal_corrected` / `model_proposal_unverified`). Invisible points are marked invisible, not guessed.
2. **Behaviour: interval table per track** (start/end time, not per frame):

| Field | Values |
| --- | --- |
| session_id, camera_id, track_id | link to tracks |
| layer | `activity` \| `indicator` \| `scenario_phase` |
| label | activity: swim_stroke, tread_water, float_back, float_face_down, rest_edge, stand, dive, underwater_swim, play, other · indicator: vertical_posture, head_tilted_back, head_submerged, ineffective_arm_motion, no_forward_progress, repeated_submersion · scenario_phase (C only): normal, onset, distress_simulated, recovery, assisted |
| start_s, end_s | interval |
| onset_earliest_s, onset_latest_s | uncertainty window for transitions |
| confidence | certain / likely / uncertain |
| reviewer_id, review_round | who and when |
| decision | accepted / disputed / adjudicated |

Transitions [R]: label onset as a **window** (earliest-latest), not a frame. In C, the protocol log stores the
**scripted** onset (the signal given to the actor). Reviewers label the **observed** onset blind to the script.
The difference between the two is itself a result: how long before distress becomes visible.

Disagreement [R]: two independent reviewers per interval; agreement measured (interval IoU, Cohen's kappa on
labels); disagreements adjudicated by a third person or kept as `uncertain`. Nothing ambiguous is forced into a
distress label, and the word "drowning" is not used as a label for any simulated footage.

## 7. Safe collection protocol (outline, to be approved by the university ethics committee) [R]

- **Approval first:** ethics committee approval and the pool operator's written permission before any filming.
- **Closed sessions:** film only consenting adult participants in reserved lanes or hours. No bystanders in frame,
  or crop/blur them.
- **Consent:** written informed consent covering filming, annotation, storage, retention period, use in the thesis
  and any publication, and the right to withdraw (with deletion).
- **Safety (C):** a qualified lifeguard on deck and one in the water for every run; shallow-to-standing depth for
  distress scenarios; agreed stop signal; short runs with rest; no breath-holding beyond normal swimming; no
  instruction to sink or stay under; the safety lead can stop any run. Scenarios are written and reviewed by the
  safety personnel, not improvised.
- **Privacy:** raw video on restricted, encrypted storage (not GitHub, not public Drive links); participant ids
  instead of names; blurred faces in anything shown outside the team (pose annotation needs the raw head region,
  so it happens on restricted copies); deletion at the end of the retention period.
- **Data protection law:** follow the law that applies at the filming location and the university's policy
  (not assessed here).

## 8. Training and evaluation stages

| Stage | Data | Method [R] | Metrics |
| --- | --- | --- | --- |
| P1 pose baseline | real pose val (B/C frames + existing 150 `real_val`) | zero-shot: ViTPose-L SwimXYZ, COCO ViTPose-L, YOLO11-pose | PCK@0.05/0.1/0.2 upper body, OKS AP, share of frames with >= 3 confident upper-body points, latency; per subset (upright, horizontal, occluded, multi-swimmer, per camera view) |
| P2 pose adaptation | A (sampled) + real train frames | fine-tune from the SwimXYZ checkpoint, fixed real share per batch, model selection on real val only | same as P1, plus a check that the upright-posture subset did not get worse |
| B1 normal model | B sequences (tracks + pose + motion features) | describe normal behaviour per activity; first rules and simple statistics, learned models only with enough sessions | false indicator rate per hour on held-out normal sessions |
| B2 indicators | B + C interval labels | per-indicator detectors (rule thresholds first, then small temporal models) | interval-level precision/recall, onset error (s) |
| R1 risk replay | full sessions from B and C | replay through the existing FSM offline, with and without pose features (ablation) | **false alerts per hour** (B), **time-to-alert** and event recall on simulated distress (C), alert level reached; reported per session |
| R2 shadow mode | live or recorded new sessions | system runs, lifeguards review, no production weight change | same as R1, plus lifeguard agreement |

Pose features keep **risk weight 0** until R1 and R2 show they lower false alarms or time-to-alert without
increasing the other [V rule, R procedure]. Pose alone can never raise ALARM (test required).

Why rules before learned models at B1/B2 [R]: with tens of sessions and a few hundred simulated events, a learned
classifier will overfit to the actors and pools. Interpretable rules (for example vertical posture + no forward
progress + duration) are easier to validate and to explain in the thesis. Our earlier temporal model already showed
this: 6/6 caught, but 12-30 of 35 swimmer tracks raised false alerts [V].

## 9. Integration with the existing architecture

Justified changes only [R]:

1. `PoseEstimator` interface and an optional pose step after tracking (as planned); keypoints fill the existing
   `TrackObs.keypoints` field, so the **contract does not change** [V].
2. Pose features appended to `feature_spec.json` as a minor version (planned `body_verticality`, `arm_pattern` [V]).
3. **New: an offline session-replay evaluator** that runs recorded sessions through the full pipeline and computes
   false alerts per hour and time-to-alert against the interval labels. This is the most important addition,
   because every later decision depends on it.
4. An importer for the interval-label table (read-only, versioned schema).

Not justified now: changes to the FSM, the alert logic or the drone path.

Two notes on the stated architecture: our records say the detectors are **YOLO11s** fine-tuned models (Z5cc,
ddt), not YOLO11n [V `PROJECT_PROGRESS_SUMMARY.md`]; and the MAVLink path was tested against a fake connection,
not ArduPilot SITL [V same file]. Please correct the proposal text if those records are out of date.

## 10. Effort estimate [A]

| Item | Assumption | Effort |
| --- | --- | --- |
| Ethics approval and pool permission | university process | 2-6 weeks calendar |
| B filming | 6-10 sessions, 2 pools, 1-2 h each | 10-20 h filming |
| C filming | 8-12 actors x 10-15 scripted runs x 1-2 min | 3-5 sessions |
| Pose annotation | correcting proposals, about 1-2 min per person-frame | 1,000 frames: 20-35 h |
| Interval labels | 2 reviewers, about 2-4x video duration each | 10 h of video: 40-80 h total |
| Engineering | replay evaluator, label importer, pose step | 2-4 weeks |

## 11. Risks, limitations and alternatives

- **Acting validity (main risk):** mitigate by writing scenarios with lifeguards who have seen real incidents,
  by reporting results as "simulated", and by relying on the submersion timer for the passive case.
- **Actor leakage:** the same actor in train and test inflates results; split by participant.
- **Too little C data for learning:** treat C as a **test set** first, not training data. Alternative: train only on
  normal data (B) and treat departures from normal as evidence (anomaly detection), so distress data is needed
  only for evaluation.
- **Domain shift between pools and cameras:** hold out one pool.
- **Synthetic dominance (A):** fixed real share per batch, real-only model selection.
- **Scope risk for a graduation timeline:** if time is short, the strongest minimal version is B + the replay
  evaluator + the existing rules, with pose as an evaluated add-on. C can then serve as a small, clearly labelled
  demonstration set.
