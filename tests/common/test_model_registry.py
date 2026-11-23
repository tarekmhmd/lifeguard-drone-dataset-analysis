"""M2-12 model registry: checksums, naming, fetch, and start-up refusal with a SYSTEM FAULT."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from lifeguard.ai.config import load_ai_config
from lifeguard.ai.perception.factory import create_engine
from lifeguard.ai.perception.mock_engine import MockPerceptionEngine
from lifeguard.app.dashboard_controller import DashboardController, SourceChoice
from lifeguard.common.models import ModelIntegrityError, ModelRegistry, integrity_fault, sha256_file
from lifeguard.contracts.enums import HealthComponent, Status
from lifeguard.contracts.errors import ConfigError
from lifeguard.system.geometry.stub import StubPoolGeometry

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "models" / "registry.yaml"
FIXTURE = ROOT / "tests" / "fixtures" / "synthetic" / "submersion_seed0.jsonl"
SYS = ROOT / "configs" / "system.yaml"


def _registry(tmp_path: Path, file: Path, sha: str | None = None, name: str = "fixture_mock_test_v1",
              url: str | None = None) -> Path:
    entry = {"name": name, "version": "1.0.0", "kind": "mock_fixture", "path": file.name,
             "sha256": sha or sha256_file(file), "mock": True}
    if url:
        entry["url"] = url
    path = tmp_path / "registry.yaml"
    path.write_text(yaml.safe_dump({"registry_version": 1, "models": [entry]}), encoding="utf-8")
    return path


def _ai_config(tmp_path: Path, fixture: Path, registry: Path) -> Path:
    path = tmp_path / "ai.yaml"
    path.write_text(f'engine: mock\nmock:\n  fixture_path: "{fixture.as_posix()}"\n'
                    f'model_registry: "{registry.as_posix()}"\n', encoding="utf-8")
    return path


def test_committed_registry_is_valid_and_all_mock_fixtures_verify() -> None:
    registry = ModelRegistry.load(REGISTRY)
    assert registry.models and all(m.mock for m in registry.models)
    assert dict(registry.report()) == {m.name: "ok" for m in registry.models}
    assert registry.verify_file(FIXTURE).name == "fixture_mock_submersion_v1"


def test_naming_convention_and_unique_names(tmp_path: Path) -> None:
    good = _registry(tmp_path, FIXTURE)
    data = yaml.safe_load(good.read_text(encoding="utf-8"))
    data["models"][0]["name"] = "SubmersionFixture"
    good.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ConfigError):
        ModelRegistry.load(good)
    data["models"][0]["name"] = "fixture_mock_a_v1"
    data["models"].append(dict(data["models"][0]))
    good.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ConfigError):
        ModelRegistry.load(good)


def test_tampered_missing_and_unregistered_files_are_refused(tmp_path: Path) -> None:
    copy = tmp_path / "fixture.jsonl"
    shutil.copy(FIXTURE, copy)
    registry = ModelRegistry.load(_registry(tmp_path, copy))
    registry.verify_file(copy)
    copy.write_bytes(copy.read_bytes() + b"\n")
    with pytest.raises(ModelIntegrityError, match="checksum mismatch"):
        registry.verify_file(copy)
    assert registry.report() == [("fixture_mock_test_v1", "MISMATCH")]
    copy.unlink()
    with pytest.raises(ModelIntegrityError, match="file missing"):
        registry.verify("fixture_mock_test_v1")
    with pytest.raises(ModelIntegrityError, match="not registered"):
        registry.verify_file(FIXTURE)
    with pytest.raises(ModelIntegrityError, match="not in the registry"):
        registry.verify("detector_yolo11n_coco_v1")


def test_fetch_verifies_before_installing(tmp_path: Path) -> None:
    source = tmp_path / "remote" / "model.bin"
    source.parent.mkdir()
    source.write_bytes(b"weights v1")
    target_dir = tmp_path / "models"
    target_dir.mkdir()
    reg_path = _registry(target_dir, source, url=source.as_uri())
    data = yaml.safe_load(reg_path.read_text(encoding="utf-8"))
    data["models"][0]["path"] = "model.bin"
    reg_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    installed = ModelRegistry.load(reg_path).fetch("fixture_mock_test_v1")
    assert installed.read_bytes() == b"weights v1"

    installed.unlink()
    source.write_bytes(b"tampered")
    with pytest.raises(ModelIntegrityError, match="not installed"):
        ModelRegistry.load(reg_path).fetch("fixture_mock_test_v1")
    assert not installed.exists()
    assert list(target_dir.glob("*.part")) == []


def test_factory_verifies_mock_fixture(tmp_path: Path) -> None:
    copy = tmp_path / "fixture.jsonl"
    shutil.copy(FIXTURE, copy)
    cfg_path = _ai_config(tmp_path, copy, _registry(tmp_path, copy))
    assert isinstance(create_engine(load_ai_config(cfg_path), "cam_a", "s1"), MockPerceptionEngine)
    copy.write_bytes(copy.read_bytes().replace(b'"t_capture"', b'"t_capture" ', 1))
    with pytest.raises(ModelIntegrityError):
        create_engine(load_ai_config(cfg_path), "cam_a", "s1")


def test_unregistered_weights_refused_before_loading(tmp_path: Path) -> None:
    weights = tmp_path / "unknown.pt"
    weights.write_bytes(b"not a model")
    cfg = tmp_path / "ai.yaml"
    cfg.write_text(f'engine: yolo\nyolo:\n  weights: "{weights.as_posix()}"\n'
                   f'model_registry: "{_registry(tmp_path, FIXTURE).as_posix()}"\n', encoding="utf-8")
    with pytest.raises(ModelIntegrityError, match="not registered"):
        create_engine(load_ai_config(cfg), "cam_a", "s1", geometry=StubPoolGeometry())


def test_models_by_name_need_a_registry(tmp_path: Path) -> None:
    cfg = tmp_path / "ai.yaml"
    cfg.write_text("engine: yolo\nyolo:\n  model: detector_yolo11n_coco_v1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="model_registry"):
        load_ai_config(cfg)
    cfg.write_text("engine: yolo\nyolo:\n  model: a\n  weights: b.pt\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="exactly one"):
        load_ai_config(cfg)


def test_integrity_fault_is_a_system_error() -> None:
    fault = integrity_fault(ModelIntegrityError("checksum mismatch"))
    assert fault.component == HealthComponent.SYSTEM and fault.status == Status.ERROR
    assert "checksum mismatch" in fault.detail


def test_main_refuses_to_start_with_system_fault(tmp_path: Path) -> None:
    copy = tmp_path / "fixture.jsonl"
    shutil.copy(FIXTURE, copy)
    cfg = _ai_config(tmp_path, copy, _registry(tmp_path, copy, sha="0" * 64))
    proc = subprocess.run([sys.executable, "-m", "lifeguard.app.main", "--ai-config", str(cfg),
                           "--system-config", str(SYS), "--synthetic"],
                          cwd=ROOT, capture_output=True, text=True, timeout=120, check=False)
    assert proc.returncode == 3
    assert "[ALARM] SYSTEM FAULT (system): model integrity check failed" in proc.stdout


def test_dashboard_shows_system_fault(tmp_path: Path) -> None:
    copy = tmp_path / "fixture.jsonl"
    shutil.copy(FIXTURE, copy)
    cfg = _ai_config(tmp_path, copy, _registry(tmp_path, copy, sha="0" * 64))
    ctl = DashboardController()
    assert not ctl.start(cfg, SYS, SourceChoice("synthetic"))
    assert ctl.error is not None and ctl.error.startswith("SYSTEM FAULT")


def test_registry_cli(tmp_path: Path) -> None:
    def cli(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, "-m", "lifeguard.common.models", *args], cwd=ROOT,
                              capture_output=True, text=True, timeout=60, check=False)

    ok = cli("verify")
    assert ok.returncode == 0 and "ok        fixture_mock_submersion_v1" in ok.stdout
    assert cli("hash", str(FIXTURE)).stdout.strip() == sha256_file(FIXTURE)
    copy = tmp_path / "fixture.jsonl"
    shutil.copy(FIXTURE, copy)
    reg = _registry(tmp_path, copy, sha="0" * 64)
    assert cli("verify", "--registry", str(reg)).returncode == 3
