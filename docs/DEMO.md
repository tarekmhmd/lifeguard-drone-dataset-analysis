# Demo guide (supervisor presentation)

What the demo shows, end to end on this laptop (GTX 1650): the pool-trained detector (Z5cc, YOLO11s)
finding and tracking swimmers in real pool video it never saw in training, the risk engine raising an
alarm, the lifeguard confirming it in the dashboard, and the **simulated** drone response.

## Prerequisites (already set up on the demo laptop)

- Python env with GPU: `E:\lifeguardDron\.venv-train` (torch CUDA, ultralytics, streamlit, this repo
  installed editable).
- Weights: `models/z5cc_s448_final.pt` (git-ignored; backup in
  `E:\lifeguardDron\Z5cc_work_backup\Z5cc_work\runs\z5cc_v2_final\weights\last.pt`).
- Demo videos: `E:\lifeguardDron\demo\demo_NN_<swim|drown>_<video>.mp4`, one per **test** clip
  (videos never used for training), upscaled ×3 from the 224 px dataset crops.
  Rebuild with `E:\lifeguardDron\demo\build_demo.py`.

## Start the dashboard

```powershell
cd E:\lifeguardDron\lifeguard-system
E:\lifeguardDron\.venv-train\Scripts\python.exe -m streamlit run src/lifeguard/app/dashboard.py
```

The browser opens at http://localhost:8501.

## Scenario 0 — near-drowning while still visible: early PRE_ALERT (≈ 21 s) ⭐

Same sidebar settings as scenario 1, video `E:\lifeguardDron\demo\demo_11_drown_iVMA7GhSJjM.mp4`
(a swimmer struggling at the surface; they never submerge, so the submersion rule cannot fire).
**Type every field and press Enter**, otherwise Streamlit keeps the old value (with the old AI config the
mock engine runs and reports a frame-size fault).

About 2 s after the swimmer appears, the box turns orange and a **PRE_ALERT** card appears
(reason `DETECTOR_DISTRESS`: the detector's "drowning" confidence averaged over 2 s reached 0.5).
Enter a lifeguard name, press **Confirm emergency** (PRE_ALERT → ALARM), then **Confirm** again
(ALARM → DISPATCH): `[SIMULATION] mission-0001 started`.

Measured on the 12 test clips (2026-11-25, threshold 0.5 chosen before the run):

| Clips | Detector-distress PRE_ALERT |
| --- | --- |
| 6 near-drowning clips | **5 / 6** caught, after 1.9–8.0 s (missed: `demo_09`, drowning starts mid-clip) |
| 6 normal swimming clips | **0 / 6** false PRE_ALERT |

Twelve clips are a small sample: this is evidence, not a validation.

## Scenario 1 — drowning clip, alarm and simulated drone (≈ 20 s)

Sidebar:

| Field | Value |
| --- | --- |
| AI config | `E:\lifeguardDron\lifeguard-system\configs\ai_z5cc.yaml` |
| System config | `E:\lifeguardDron\lifeguard-system\configs\system_demo_drone.yaml` |
| Input | video |
| Video file | `E:\lifeguardDron\demo\demo_06_drown_T5mDQeDkca0.mp4` |
| Lifeguard name | any name (press Enter) |

Press **Start**. Boxes and track IDs appear on every swimmer. After about 17 s the swimmer has been
out of sight long enough for the submersion rule: an **ALARM** card appears. Press **Confirm emergency**:
the incident goes to DISPATCH and the response panel shows
`[SIMULATION] mission-0001 started, hover at (x, y)` — the drone mission planner, in simulation.

## Scenario 2 — normal swimming, no alarm (≈ 12 s)

Same settings, video `E:\lifeguardDron\demo\demo_05_swim_L0KTqPloUiU.mp4`: swimmers are tracked,
no incident is raised.

## Scenario 3 (optional) — full synthetic scenario

Input **synthetic** with AI config `configs/ai.yaml`: the mock engine replays a scripted scenario through
the whole stack (no model). Useful to show every risk level and the operator workflow.

## Results to show next to the demo

- `E:\lifeguardDron\Z5cc_work_backup\Z5cc_work\RESULTS_test.csv` — all experiments.
- `...\runs\z5cc_v2_final_test\confusion_matrix_normalized.png`, `BoxPR_curve.png`.
- `...\runs\z5cc_v2_s448\results.png` — training curves.

| Test split (6 unseen videos) | baseline n224 | final s448 (train+val) |
| --- | --- | --- |
| mAP50 | 0.571 | **0.704** |
| drowning AP50 | 0.443 | **0.649** |
| drowning recall / precision | 0.509 / 0.509 | **0.633 / 0.737** |

## Honest limitations (say them before they are asked)

- **Pool geometry is a placeholder** (no calibration of these YouTube clips). Pool positions are
  approximate and a dark object at the image edge can be taken for a person; a calibrated camera
  masks everything outside the water.
- **The clips are 224 px crops lasting 10–40 s.** Swimmers leaving the crop look like submersions, so
  two swimming clips (`demo_01`, `demo_07`) also raise a submersion alarm. Alarm rates must be measured
  on full-pool recordings, not on these crops.
- **The detector's drowning class never raises an ALARM by itself.** Its 2 s mean can raise a
  PRE_ALERT ("check this swimmer") when `risk.detector_distress_threshold` is set (only in
  `system_demo_drone.yaml`; off in `system.yaml`). ALARM comes from submersion timing or a lifeguard
  confirmation; the trained temporal model is the next step (`docs/TEMPORAL_PLAN.md`).
- **The drone is simulation only.**

## Headless check (no browser)

```powershell
cd E:\lifeguardDron\lifeguard-system
E:\lifeguardDron\.venv-train\Scripts\python.exe -m lifeguard.app.main --ai-config configs/ai_z5cc.yaml `
    --system-config configs/system.yaml --video E:\lifeguardDron\demo\demo_06_drown_T5mDQeDkca0.mp4
```

Measured on 2026-11-25: 574 frames, 0 errors, about 14 ms per frame, 1 alarm (submersion timeout).
