# Autonomous AI Lifeguard System

Swimming-pool lifeguard *assistant*: video → person detection and tracking → temporal risk
assessment → incidents and alerts for lifeguards → (optional, simulation-only) drone response.
It raises suspected incidents for humans to check; it never declares a drowning and never acts
physically on its own.

**Status:** the software is complete and tested on CPU without cameras or model weights. It has
not been verified with real cameras or ArduPilot SITL, and its AI has not been validated on labelled
pool recordings. See [`docs/SYSTEM_GUIDE.md`](docs/SYSTEM_GUIDE.md) for the architecture, what is
real/mock/simulated, configuration and all commands, and [`docs/ROADMAP.md`](docs/ROADMAP.md) for
history and open decisions.

Try the whole system without a camera or weights (mock perception, everything else real):

```powershell
python -m lifeguard.app.main --ai-config configs/ai.yaml --system-config configs/system.yaml --synthetic
python -m pip install -e ".[dashboard]"; streamlit run src/lifeguard/app/dashboard.py
```

Repository areas:

| Path | Contents |
| --- | --- |
| `src/lifeguard/` | Python package: `contracts` (frozen v1.0), `ai` (perception, tracking, behaviour), `system` (risk, alerts, health, recording, geometry, ingest, response), `robotics` (simulated drone), `app` (runtime, CLIs, dashboard) |
| `tests/` | pytest suite (unit, `integration/` milestone tests), contract examples and generated synthetic fixtures |
| `configs/` | `ai.yaml` / `ai_yolo.yaml` (Member 1) and `system.yaml` (Member 2) |
| `models/` | `registry.yaml`: every loadable model with its sha256 (weights themselves are never committed) |
| `docs/` | [`SYSTEM_GUIDE.md`](docs/SYSTEM_GUIDE.md), [`index.md`](docs/index.md) (decisions, workflow, milestones), [`ai/annotation_guide.md`](docs/ai/annotation_guide.md), [`ROADMAP.md`](docs/ROADMAP.md), dataset decision reports |
| `python_scripts/`, `notebooks/` | Dataset and training scripts (see `python_scripts/README.md`) |

## Phase 1 developer quickstart

Requires Python 3.11 (3.10+ supported). Commands are shown for Windows PowerShell; on
Linux/macOS activate with `source .venv/bin/activate`.

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install the package with developer tools

```powershell
python -m pip install -e ".[dev]"
```

### 3. Verify (the same checks CI runs)

```powershell
python -m ruff check src tests
python -m pytest
python -m lifeguard.ai.sim.generate_fixtures --out tests/fixtures/synthetic --check
python -m lifeguard.app.replay_mock --ai-config configs/ai.yaml --system-config configs/system.yaml
```

Expected results:

- `ruff` reports no errors.
- `pytest` reports 0 failures and 0 errors.
- `generate_fixtures --check` exits 0 (committed fixtures match the generator).
- `replay_mock` exits 0 and prints one line of the form
  `frames=<N> ok=<n> degraded=<n> error=<n> unique_tracks=<k> max_submersion_s=<x.xxx>`,
  with `max_submersion_s` > 0 for the default `submersion` fixture.

`replay_mock` options: `--frames N` (default: fixture length) and `--out PATH.jsonl`
(write every `FrameResult`). Exit codes: 0 success, 2 bad arguments/configuration,
3 model/fixture load error, 4 too many consecutive engine errors.

Running the real pipeline on a recorded video (optional; needs `pip install -e ".[vision]"` and a local
weights file referenced by `configs/ai_yolo.yaml`; pool geometry is a placeholder rectangle until
calibration exists, and nothing here is validated on pool video):

```powershell
python -m lifeguard.app.run_video --ai-config configs/ai_yolo.yaml --system-config configs/system.yaml --video path\to\pool.mp4 --out-events events.jsonl
```

Event-level evaluation of the risk engine (missed events, false alarms per hour, latency; on the
synthetic fixtures this checks logic only, not real-world accuracy):

```powershell
python -m lifeguard.app.evaluate --frames tests/fixtures/synthetic/submersion_seed0.jsonl --truth tests/fixtures/synthetic/submersion_seed0_events.csv --system-config configs/system.yaml --sweep t_sub_s=6,8,10,12
```

To regenerate the synthetic fixtures after an intentional generator change:

```powershell
python -m lifeguard.ai.sim.generate_fixtures --out tests/fixtures/synthetic --seed 0
```

### Phase 1 documentation

- [`docs/phase1/PHASE_1_IMPLEMENTATION_SPEC.md`](docs/phase1/PHASE_1_IMPLEMENTATION_SPEC.md)
- [`docs/phase1/PHASE_1_TEST_PLAN.md`](docs/phase1/PHASE_1_TEST_PLAN.md)
- [`docs/phase1/PHASE_1_EXECUTION_PLAN.md`](docs/phase1/PHASE_1_EXECUTION_PLAN.md)
- [`docs/phase1/PHASE_1_DECISIONS_AND_ASSUMPTIONS.md`](docs/phase1/PHASE_1_DECISIONS_AND_ASSUMPTIONS.md)
- [`docs/phase1/PHASE_1_HANDOFF_CHECKLIST.md`](docs/phase1/PHASE_1_HANDOFF_CHECKLIST.md)
- [`docs/phase1/CODING_AGENT_INSTRUCTIONS.md`](docs/phase1/CODING_AGENT_INSTRUCTIONS.md)

> Note: the `docs/phase1/` documents are not yet part of this repository; they currently
> live in the main project workspace and must be copied into `docs/phase1/` for these links
> to resolve.
