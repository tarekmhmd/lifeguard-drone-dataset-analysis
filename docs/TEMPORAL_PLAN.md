# Temporal behaviour layer — training plan (P5-2)

Status: **FIRST TRAINING DONE, NOT ADOPTED** (2026-11-25, section 8). The code in `lifeguard.ai.temporal` is
implemented and tested; a model was trained and evaluated but does not beat the detector-distress rule. This document explains how the layer is trained from the Z5cc pool videos once the pool detector
(P4-7) is final.

## 1. Why a temporal layer

The YOLO detector looks at **one frame**. Drowning is a behaviour that unfolds over **seconds**: little
forward progress, vertical body, bobbing, intermittent disappearance (the Instinctive Drowning Response
described by F. Pia). The temporal layer follows the same tracked person for 2–3 s and estimates the
probability of drowning from that behaviour, not from a single image.

The detector's per-frame drowning score is used as one input feature, never as an alarm on its own.

## 2. Pipeline

```
① final Z5cc detector (best.pt)
        │
② full Z5cc clips (z5cc_v1.zip: 91 clips, 30 fps, every frame)
        │  run the real engine: YOLO → ByteTrack → FeatureExtractor
        ▼
③ frames.jsonl   per track and frame: motion features (+ detector drowning score)
        +
④ events.csv     from the Z5cc labels: "person X drowning from t=12 s to t=19 s"
        │
⑤ ai.temporal.dataset   one sample per track every 0.5 s, labelled from the events
        │
⑥ ai.temporal.train     softmax model, leave-one-session-out cross-validation
        ▼
⑦ models/behavior_softmax.json
```

## 3. One sample

Every 0.5 s, for every in-water track:

| Feature | Meaning | Swimming | Drowning |
| --- | --- | --- | --- |
| `speed_mps` / `path_speed_mps` | speed | high | near zero |
| `progress_ratio` | forward progress vs. moving in place | high | low |
| `bob_std_norm` | vertical bobbing | low | **high** |
| `visibility_ratio` | how much of the window the person was visible | high | drops |
| `time_since_seen_s` | time since last detection | 0 | grows |
| `box_aspect` | box shape | wide (horizontal body) | **tall** (vertical body) |
| `yolo_drown` (new) | mean detector drowning score over the last 2 s | low | high |

Label: `drowning` while the time is inside a drowning event for that person, otherwise the default
`normal_swim`.

## 4. Work items

1. **Z5cc labels → `events.csv`.** Each run of consecutive frames in which a person is labelled
   `drowning` becomes one `GroundTruthEvent` (DROWNING, onset, end, position from the box centre).
2. **Detector distress feature — DONE (2026-11-25).** `Detection.distress` carries the confidence of
   the model's distress class (`yolo.distress_class_ids`, Z5cc `[1]`) through the tracker; the real
   engine publishes the per-track 2 s mean as `TrackObs.features["det_distress_2s"]`
   (`contracts.types.DISTRESS_FEATURE`). The risk engine already uses it for a PRE_ALERT rule
   (`risk.detector_distress_threshold`). It is not part of `feature_spec.json` (the extractor does not produce it); adding it to the spec is a minor
   spec version and only worth doing once a trained model uses it.
3. **Pixel-normalised motion features.** The Z5cc clips are 224 px crops without pool calibration, so
   speeds in m/s are unavailable. Use speeds normalised by the person's box size (body lengths per
   second) when no calibration is configured.
4. **Keep the video split (no leakage).** Train the temporal model on train-split clips only and report
   on test-split clips only. The detector was trained on the train videos, so its drowning score is
   over-confident there; a temporal model must not be evaluated on those clips.

## 5. Integration and acceptance

- `configs/ai_yolo.yaml`: `behavior: softmax`, `behavior_model: ../models/behavior_softmax.json`.
- `risk.weights.behavior` **stays 0** until the evaluation below passes (P5-3).
- Evaluate with `python -m lifeguard.app.evaluate` on the test clips, comparing:
  1. rules only (current risk engine),
  2. rules + temporal model,
  3. detector frame-by-frame drowning score.
- Metrics: event recall (drowning episodes caught), time to alarm, false alarms per hour.
- Accept (P5-3) only if rules + temporal catches more episodes at equal or fewer false alarms per hour.

## 6. Expected result to report

A drowning episode spans many frames, so an event-level detector can catch an episode even when the
frame-level detector misses some of its frames. The report should state both numbers: frame-level drown
recall of the detector (Z5cc s448: 0.58 on the test split) and event-level recall / time to alarm of the
full system.

## 7. Related work (verify citations before use)

- F. Pia, *Instinctive Drowning Response* (1974): the behavioural signs used by `rules.py`.
- H.-L. Eng et al., DEWS, IEEE (~2008): overhead pool cameras, motion features and a classifier; same
  overall design (detect → track → features → classify → alarm).
- Commercial systems (Poseidon, SwimEye, Lynxight): motionless-underwater timer, which corresponds to the
  risk engine's submersion term and hard rule.

## 8. Result of the first training (2026-11-25)

The experiment ran in the development copy (`E:\lifeguard-github-upload\python_scripts\z5cc_temporal_experiment.py`, stages `extract`, `train`, `eval`) on the feature layout that existed there before
`feature_spec.json`; it is not part of this repository because the model was not adopted. Instead of
`events.csv`, every sample is labelled from the annotated box overlapping the track in that frame
(IoU >= 0.3), which is exact for Z5cc.

- **Features:** the real engine on the continuous Z5cc clips (30 fps, placeholder pool filling each 224 px crop).
  For the experiment, `det_distress_2s` was appended to the model vector.
- **Training set: val clips only** (14 clips, 661 samples every 0.5 s: 423 normal_swim, 238 drowning), with the
  **s448** detector (trained on train only), so the detector feature is realistic. The final detector saw
  train+val, so its scores on those clips would be over-optimistic.
- **Test:** 12 test clips with the deployed final detector. Model file (development copy only): `models/behavior_softmax_z5cc.json`.

Sample level (804 test samples, 143 drowning):

| Signal | threshold | precision | recall |
| --- | --- | --- | --- |
| detector 2 s mean (`det_distress_2s`) | 0.5 | **1.00** | 0.62 |
| softmax, all features | 0.5 / 0.7 | 0.57 / 0.69 | 0.83 / 0.78 |
| softmax, not class-weighted | 0.5 / 0.7 | 0.67 / 0.75 | 0.79 / 0.74 |
| softmax, **motion features only** (no detector) | 0.5 / 0.7 | 0.25 / 0.13 | 0.48 / 0.09 |

Episode level (6 drowning episodes, 35 swimmer tracks >= 2 s):

| Alert rule | episodes caught | median latency | swimmer tracks with a false alert |
| --- | --- | --- | --- |
| **A detector 2 s mean >= 0.5 (deployed PRE_ALERT rule)** | **5 / 6** | 3.6 s | **0 / 35** |
| softmax p >= 0.5 | 6 / 6 | 0.1 s | 30 / 35 |
| softmax 2 s mean >= 0.7 | 6 / 6 | 3.3 s | 12 / 35 |
| A or softmax 2 s mean >= 0.7 | 6 / 6 | 3.1 s | 12 / 35 |

**Conclusion: the trained model does not beat the detector rule, so it is NOT enabled** (`behavior: none`,
risk weight 0). The learned weights point the right way (drowning = low forward progress, more bobbing,
lower visibility, higher detector score). But motion features alone are close to chance on this data
(precision 0.25 against a 0.18 base rate). Reasons:

- The clips are 224 px crops of different pools with no calibration, so speeds are not comparable between clips.
- Many swimmers are static (on floats, treading water) and look like the instinctive drowning response.
- Training used only 14 clips.

What would change this: full-frame video from a calibrated camera, so motion is measured in metres, and more
labelled sessions (staged recordings). The pipeline (features → samples → model → evaluation) is ready to
re-run on such data.
