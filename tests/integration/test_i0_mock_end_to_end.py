"""Milestone I0 — mock end to end (SYSTEM_INTEGRATION_CHECKLIST.md section 2).

Exit criteria checked here:
- the submersion fixture produces an ALARM (dashboard state), a saved clip and an incident-log row;
- the dive, edge_exit and normal_swim fixtures produce no ALARM;
- confirm and dismiss both work, and the dismissal is logged.
(The remaining criterion, "contract tests green in CI, tag v0.2-I0", needs a push and a tag.)
Perception is the MOCK fixture replay (registered in models/registry.yaml); everything downstream is real.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from lifeguard.ai.config import load_ai_config
from lifeguard.app.runtime import LifeguardRuntime
from lifeguard.contracts.enums import AckDecision, RiskLevel
from lifeguard.system.config import load_system_config
from lifeguard.system.ingest.live import SyntheticFrameSource

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "synthetic"
REGISTRY = ROOT / "models" / "registry.yaml"
T0 = "2026-01-01T00:00:00.000Z"
LENGTH_S = {"submersion": 60, "dive": 40, "edge_exit": 40, "normal_swim": 60}


def _run(kind: str, tmp_path: Path) -> LifeguardRuntime:
    tmp_path.mkdir(parents=True, exist_ok=True)
    ai = tmp_path / f"ai_{kind}.yaml"
    ai.write_text(f'engine: mock\nmock:\n  fixture_path: "{(FIXTURES / f"{kind}_seed0.jsonl").as_posix()}"\n'
                  f'model_registry: "{REGISTRY.as_posix()}"\n', encoding="utf-8")
    cfg = load_system_config(ROOT / "configs" / "system.yaml")
    cfg.recording.out_dir = tmp_path / f"clips_{kind}"
    cfg.alerts.incident_db = tmp_path / f"incidents_{kind}.sqlite3"
    rt = LifeguardRuntime(load_ai_config(ai), cfg, (1920, 1080), session_id=f"i0_{kind}")
    rt.run(SyntheticFrameSource(1920, 1080, n_frames=LENGTH_S[kind] * 10, start_t_capture=0.0, start_utc=T0))
    return rt


def _alarm_incident(rt: LifeguardRuntime) -> str:
    alarms = [(iid, lvl) for iid, lvl in rt.snapshot().open_incidents.values() if lvl == RiskLevel.ALARM]
    assert alarms, "no ALARM"
    return alarms[0][0]


def test_submersion_alarm_clip_and_db_row(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    rt = _run("submersion", tmp_path)
    incident_id = _alarm_incident(rt)
    assert any(n.message.startswith("ALARM:") for n in rt.snapshot().notifications)
    (row,) = [r for r in rt.store.incidents() if r["incident_id"] == incident_id]
    assert row["max_level"] == "ALARM"
    (clip,) = rt.store.clips(incident_id)
    assert Path(clip["path"]).is_file()
    rt.close()


@pytest.mark.parametrize("kind", ["dive", "edge_exit", "normal_swim"])
def test_negative_fixtures_produce_no_alarm(kind: str, tmp_path: Path) -> None:
    rt = _run(kind, tmp_path)
    assert rt.snapshot().summary.alarms == 0
    assert all(r["max_level"] != "ALARM" for r in rt.store.incidents())
    rt.close()


def test_confirm_and_dismiss_work_and_dismissal_is_logged(tmp_path: Path) -> None:
    confirm_rt = _run("submersion", tmp_path / "a")
    event = confirm_rt.alerts.latest_event_id(_alarm_incident(confirm_rt))
    result, _ = confirm_rt.acknowledge(event, AckDecision.CONFIRM, "lifeguard_1")
    assert [e.to_level for e in result.events] == [RiskLevel.DISPATCH]
    assert confirm_rt.store.dispatches()[0].confirmed_by == "lifeguard_1"
    confirm_rt.close()

    dismiss_rt = _run("submersion", tmp_path / "b")
    event = dismiss_rt.alerts.latest_event_id(_alarm_incident(dismiss_rt))
    result, _ = dismiss_rt.acknowledge(event, AckDecision.DISMISS, "lifeguard_2")
    assert [e.to_level for e in result.events] == [RiskLevel.RESOLVED]
    (negative,) = dismiss_rt.store.hard_negatives()
    assert negative["ack_by"] == "lifeguard_2" and negative["t_utc"]
    dismiss_rt.close()
