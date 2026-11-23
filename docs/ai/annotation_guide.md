# Annotation Guide — swimmer behaviour (M1-04)

Version 1.0.0 (matches `src/lifeguard/ai/temporal/label_taxonomy.json`, `taxonomy_version` 1.0.0).
Audience: everyone labelling pool video for this project. Read it fully before your first session.

## 1. Ground rules

1. **A detected person is not a drowning person.** The detector only finds people. Nothing in a single
   frame proves distress.
2. **Label behaviour over time.** Every interval has a minimum duration (table below). Never label a frame.
3. **Use what you can see.** Each label has observable criteria. Do not label from intent, outcome, the
   scenario sheet or what you expect to happen.
4. **No single cue decides `distress`.** Raised arms alone, a face close to the water alone, an unusual
   posture alone or a submersion alone are **not** distress.
5. **When in doubt, use `uncertain`.** Uncertain time is excluded from training and evaluation; a wrong
   label is worse than a missing one.
6. **Safety and ethics.** Distress and submersion scenarios are performed only by trained adult
   volunteers with lifeguards present; no scenario involves children. Raw video stays private.

## 2. Labels

| Label | Min. duration | Summary |
| --- | --- | --- |
| `normal_swim` | 2.0 s | Purposeful swimming with forward progress |
| `idle` | 2.0 s | Calm presence without swimming: standing, floating, resting, calm treading |
| `play` | 1.0 s | Voluntary recreation: splashing, jumping, games, interacting |
| `dive` | 0.5 s | Intentional entry or underwater movement, controlled resurfacing |
| `distress` | 3.0 s | Instinctive-drowning-response-like behaviour (staged by trained adults) |
| `uncertain` | 0.5 s | Cannot decide from the video |

`drowning` exists in the data contract but is **never annotated in staged sessions**. Real incidents are
recorded by the safety officer as event rows (section 4), not as intervals.

### `normal_swim`
Observable criteria:
- forward displacement is clearly visible (roughly ≥ 0.3 m/s, or crossing lane markings);
- a regular, repeating stroke and/or kick;
- the head clears the water regularly to breathe;
- the body is closer to horizontal than vertical.

Not this label: progress stops for more than 2 s (end the interval); splashing without progress (`play`).

### `idle`
- little or no displacement (roughly < 0.2 m/s);
- head stably above the water, unhurried breathing;
- relaxed movements: looking around, talking, holding the wall or a float;
- no repeated going under and coming up.

Not this label: the head repeatedly drops to the waterline with a vertical body and arms pressing down —
check `distress`; if its criteria are not all met, use `uncertain`.

### `play`
- energetic but controlled movement, often with other people or objects;
- arms may be raised, waving or splashing; the person turns towards others, laughs or calls out;
- brief voluntary submersions with controlled resurfacing are part of the activity;
- frequent direction changes, no sustained progress.

Not this label: alone, vertical, head at the waterline and arms pressing down for ≥ 3 s — check `distress`.

### `dive`
- the going-under follows a deliberate action (push-off, tuck, jump, surface dive, a breath taken);
- under water the person moves purposefully or holds still in a controlled way;
- resurfacing is controlled (no gasping struggle) and another activity resumes;
- the interval runs from the start of the deliberate action to resurfacing.

Not this label: no deliberate action before going under, or no resurfacing — use `uncertain` and, in a
staged submersion scenario, record a `submersion` event (section 4).

### `distress`
Criteria:
1. head at or below the waterline for more than half of the interval, mouth at water level, head tilted
   back when it surfaces;
2. no forward progress;
3. body vertical, no effective kick;
4. arms press down or extend laterally on the surface (not waving overhead, not stroking);
5. no voluntary communication (does not call out or wave for help).

**Decision rule:** label `distress` only when at least criteria 1–3 hold together for ≥ 3 s. One
criterion alone is never enough. Waving arms with the head well above water is signalling or play;
a submersion with controlled resurfacing is a `dive`; calm floating is `idle`.

### `uncertain`
Use whenever no other label's criteria are clearly met: occlusion, glare, the person leaves the frame,
or the cues conflict.

## 3. Interval file (behaviour labels)

One CSV per session, UTF-8, header exactly:

```
session_id,video_file,track_id,t_start,t_end,label,annotator,notes
```

- `session_id` — the session identifier used by the system for that recording.
- `video_file` — the recording's file name.
- `track_id` — the track number drawn by the system's overlay (`run_video` output / dashboard) for this
  session. Label the IDs you see; when the overlay shows a new ID for the same person, start a new interval
  with the new ID.
- `t_start`, `t_end` — seconds from the start of the video (`t_media`), `t_end > t_start`, at least the
  label's minimum duration.
- `label` — one of the six labels above; `annotator` — your initials or ID; `notes` — free text.
- Intervals of one annotator for one track must not overlap. Gaps are allowed (unlabelled time is never
  used as a negative example).

Check a file before handing it in:

```powershell
python -m lifeguard.ai.temporal.labels validate session_s01_labels.csv
```

## 4. Event file (evaluation labels)

Incidents the system must alarm on use the contract event CSV (read by the evaluator,
`python -m lifeguard.app.evaluate`):

```
event_id,session_id,video_file,type,t_onset,t_end,track_uid,pool_x,pool_y
```

`type` is `submersion`, `distress` or `drowning`; `t_end` may be empty (open-ended); give `track_uid`
when known, otherwise the pool position `pool_x,pool_y` (metres) at onset. This is a superset of the M1-04
event schema (`event_id,video_file,t_onset,t_end,type,pool_x,pool_y`).

## 5. Second annotator and agreement

- A second annotator re-labels a random 10 % of intervals (whole tracks), without seeing the first labels.
- Agreement is computed on the time both labelled, on a 0.5 s grid:

```powershell
python -m lifeguard.ai.temporal.labels agreement annotator_a.csv annotator_b.csv
```

- Report Cohen's kappa and percent agreement with the dataset. The acceptance threshold is a team
  decision to be made **before** labelling starts (open item); disagreements on `distress` are reviewed
  together and resolved by re-watching, never by majority vote alone.

## 6. From labels to training

```powershell
python -m lifeguard.app.run_video --ai-config configs/ai_yolo.yaml --system-config configs/system.yaml --video s01.mp4 --out-frames s01_frames.jsonl
python -m lifeguard.ai.temporal.train --intervals --frames s01_frames.jsonl --truth s01_labels.csv --frames s02_frames.jsonl --truth s02_labels.csv --cv --out models/behavior_softmax_staged_v1.json
```

The resulting model must beat the rules-only baseline in leave-one-session-out validation and be
registered in `models/registry.yaml` before the risk engine may give it any weight (milestone I2).
