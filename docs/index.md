# Project index: decisions, workflow and milestones

The integration checklist asks for decisions to be recorded in `docs/index.md`. This file is that record
for the code repository. Planning documents live in the main project (`E:\lifeguardDron\docs`):
`PARALLEL_DEVELOPMENT_ROADMAP.md`, `SYSTEM_INTEGRATION_CHECKLIST.md`, `MEMBER_1/2_NEXT_TASKS.md`,
`phase1/`. Code documentation: [SYSTEM_GUIDE.md](SYSTEM_GUIDE.md), [ROADMAP.md](ROADMAP.md),
[ai/annotation_guide.md](ai/annotation_guide.md).

## 1. Decisions

| ID | Decision | Source | Status |
| --- | --- | --- | --- |
| D-01 | Code lives in its own repository, separate from the data workspace (now `E:\lifeguardDron\lifeguard-system`) | Phase 1 decisions | confirmed |
| D-02 | Six risk levels NORMAL, WATCH, PRE_ALERT, ALARM, DISPATCH, RESOLVED | Phase 1 decisions | confirmed, in contract v1.0.0 |
| D-03 | Dispatch only on an explicit human CONFIRM; an unacknowledged ALARM repeats, never dispatches | Phase 1 decisions | confirmed, implemented |
| D-04 | CODEOWNERS usernames | Phase 1 decisions | **open** (template in `docs/workflow/CODEOWNERS.template`) |
| C-01 … C-20 | Contract and packaging decisions | `phase1/PHASE_1_DECISIONS_AND_ASSUMPTIONS.md` | confirmed |
| C-12 | `FeatureSpec`, `PoolGeometryConfig`, `MissionStatus` deferred to minor contract versions | Phase 1 decisions | `feature_spec.json` and the calibration JSON exist with their owners; promotion into the contract is **open** (needs both members) |
| I-01 | Submersion re-association radius 2.0 m (Blueprint: 1.5 m) to allow one frame of tracker confirmation | implementation (batch 2) | in `FeatureConfig`, tune on staged clips |
| I-02 | Behaviour evidence weight 0 in the risk score until a model passes milestone I2 | safety design | in `configs/system.yaml` |
| I-03 | Response platform default `none` (fixed camera / alert only); drone is simulation-only | supervisor decision pending | in `configs/system.yaml` |
| I-04 | Simulated flotation release waits for a human confirmation (`require_release_confirmation: true`) | architecture §16 | in `configs/system.yaml` |
| I-05 | Every model file is loaded only if registered with a matching sha256 (`models/registry.yaml`); otherwise SYSTEM FAULT, no start | M2-12 | implemented |
| I-06 | SYSTEM FAULT for perception after `max_consecutive_engine_errors` (10) error frames in a row; fewer = DEGRADED | integration checklist §1 | implemented |
| I-07 | Feature spec 1.0.0 (8 features, 4 s window, 0.5 s step); head visibility, verticality, arm pattern and motion energy not implemented (need data/pose/latency budget) | M1-03 | implemented |
| I-08 | Behaviour labels: `normal_swim`, `idle`, `play`, `dive`, `distress`, `uncertain`; `drowning` never annotated in staged sessions | M1-04 | implemented |
| OD-1 | Escalate an unacknowledged PRE_ALERT to ALARM after a timeout? | ROADMAP | **open** (disabled) |
| OD-6 | Acceptable false alarms per hour and latency budget | ROADMAP | **open** |
| OD-7 | Ultralytics is AGPL-3.0 | ROADMAP | **open** |
| OD-10 | Annotator agreement threshold (kappa) for accepting labels | M1-04 | **open** — decide before labelling |
| OD-11 | Detector acceptance targets (J-05) | roadmap | **open** — before final training |

## 2. Workflow conventions

- Branches: `m1/<topic>` (Member 1), `m2/<topic>` (Member 2), `int/<milestone>` (integration); `main` is
  protected once the repository is on GitHub with reviewers.
- Contract changes (`src/lifeguard/contracts/`) and `tests/integration/` need both members' approval.
  Optional new field = minor version; rename/removal = major version.
- Every merge to `main`: `python -m ruff check src tests`, `python -m pytest`,
  `python -m lifeguard.common.models verify`, `python -m lifeguard.ai.sim.generate_fixtures --out tests/fixtures/synthetic --check`
  (all run in CI). No thresholds outside `configs/`; no weights, videos or datasets in git.
- Model files: name `<task>_<arch>_<data>_v<n>`, registered in `models/registry.yaml` with sha256; weights
  stay out of git (`*.pt`, `*.onnx`, `*.engine` are ignored).
- Milestones are tagged when their exit criteria pass: `v0.2-I0`, `v0.3-I0.5`, `v0.4-I1`, `v0.5-I2`,
  `v0.6-I3`, `v0.7-I4`, `v1.0`. Tags require commits; none has been created yet.
- CODEOWNERS: copy `docs/workflow/CODEOWNERS.template` to the repository root as `CODEOWNERS` once both
  GitHub usernames are known (D-04); usernames are never guessed.

## 3. Milestone status (integration checklist)

| Milestone | Software status | What is still needed |
| --- | --- | --- |
| I0 mock end to end | exit criteria pass: `tests/integration/test_i0_mock_end_to_end.py` | CI run on GitHub, tag `v0.2-I0` |
| I0.5 real perception, first FA/h | software ready (`run_video`, `evaluate`, `mot` export) | normal pool video, real calibration, YOLO weights registered |
| I1 fine-tuned detector | registry + config swap ready | approved dataset, trained detector (M1-10), re-tuned tracker (M1-11) |
| I2 temporal evidence | labels, training, CV, evaluator ready | staged sessions (J-07), trained model (M1-13), tuning (M2-14) |
| I3 drone dispatch in simulation | pure-Python simulation passes `tests/integration/test_i3_drone_simulation.py`; MAVLink executor ready | ArduPilot SITL + Gazebo runs (M2-09) |
| I4 rescue release | simulated release with human confirmation | SITL release runs, bench rig (50 trials) |
| I5 end to end | runtime + dashboard ready | everything above |
