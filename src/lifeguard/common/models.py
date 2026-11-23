"""Model registry and integrity checks (MEMBER_2 M2-12, integration checklist section 1).

``models/registry.yaml`` lists every model file the runtime may load: detector weights, behaviour models and
the mock fixtures that stand in for them while no trained model exists. Each entry holds a name following
``<task>_<arch>_<data>_v<n>``, a version, a sha256 checksum, the file path (relative to the registry file)
and an optional download URL. Weights are never committed to git.

When ``model_registry`` is set in the AI config, the engine factory verifies every file it is about to
load. A missing entry, a missing file or a checksum mismatch raises ``ModelIntegrityError`` (a
``ModelLoadError``): the system refuses to start and reports a SYSTEM_FAULT (``integrity_fault``).

CLI::

    python -m lifeguard.common.models verify [--registry models/registry.yaml]
    python -m lifeguard.common.models hash path/to/file
    python -m lifeguard.common.models fetch NAME   # downloads the entry's URL, verifies, then installs

``fetch`` only runs when called explicitly; nothing is downloaded implicitly.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import re
import shutil
import sys
import tempfile
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from lifeguard.common.config import load_yaml_model
from lifeguard.contracts.enums import HealthComponent, Status
from lifeguard.contracts.errors import ConfigError, ModelLoadError
from lifeguard.contracts.timeutil import utc_now
from lifeguard.contracts.types import SystemHealth

logger = logging.getLogger(__name__)

NAME_PATTERN = re.compile(r"^[a-z0-9]+_[a-z0-9]+_[a-z0-9-]+_v\d+$")
DEFAULT_REGISTRY = Path(__file__).resolve().parents[3] / "models" / "registry.yaml"
_CHUNK = 1 << 20


class ModelIntegrityError(ModelLoadError):
    """A model file is unregistered, missing or does not match its registered checksum."""


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ModelEntry(BaseModel):
    """One registered model file."""

    model_config = ConfigDict(extra="forbid")

    name: str
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    kind: Literal["detector", "behavior", "mock_fixture"]
    path: Path
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    url: str | None = None
    mock: bool = False
    description: str = ""

    @field_validator("name")
    @classmethod
    def _name_convention(cls, value: str) -> str:
        if not NAME_PATTERN.match(value):
            raise ValueError(f"model name {value!r} must follow <task>_<arch>_<data>_v<n>")
        return value


class ModelRegistry(BaseModel):
    """The registry file."""

    model_config = ConfigDict(extra="forbid")

    registry_version: int = 1
    models: list[ModelEntry] = Field(default_factory=list)
    base_dir: Path | None = Field(default=None, exclude=True)

    @field_validator("models")
    @classmethod
    def _unique_names(cls, value: list[ModelEntry]) -> list[ModelEntry]:
        names = [m.name for m in value]
        if len(names) != len(set(names)):
            raise ValueError("model names must be unique")
        return value

    @classmethod
    def load(cls, path: Path | str) -> ModelRegistry:
        """Load and validate a registry. Raises ConfigError naming the file."""
        registry = load_yaml_model(path, cls)
        registry.base_dir = Path(path).resolve().parent
        return registry

    def file_of(self, entry: ModelEntry) -> Path:
        base = self.base_dir or Path.cwd()
        return entry.path if entry.path.is_absolute() else (base / entry.path).resolve()

    def get(self, name: str) -> ModelEntry:
        for entry in self.models:
            if entry.name == name:
                return entry
        raise ModelIntegrityError(f"model {name!r} is not in the registry")

    def verify(self, name: str) -> Path:
        """Check one entry; return its file. Raises ModelIntegrityError."""
        entry = self.get(name)
        return self._check(entry)

    def verify_file(self, path: Path | str) -> ModelEntry:
        """Check that ``path`` is a registered file with a matching checksum. Raises ModelIntegrityError."""
        target = Path(path).resolve()
        for entry in self.models:
            if self.file_of(entry) == target:
                self._check(entry)
                return entry
        raise ModelIntegrityError(f"{target} is not registered in the model registry; refusing to load it")

    def _check(self, entry: ModelEntry) -> Path:
        file = self.file_of(entry)
        if not file.is_file():
            raise ModelIntegrityError(f"model {entry.name}: file missing ({file})")
        actual = sha256_file(file)
        if actual != entry.sha256:
            raise ModelIntegrityError(
                f"model {entry.name}: checksum mismatch (expected {entry.sha256[:12]}..., got {actual[:12]}...)"
            )
        logger.info("model %s v%s verified (%s)", entry.name, entry.version, "mock" if entry.mock else entry.kind)
        return file

    def report(self) -> list[tuple[str, str]]:
        """``(name, status)`` for every entry: ok | missing | MISMATCH."""
        out = []
        for entry in self.models:
            file = self.file_of(entry)
            if not file.is_file():
                out.append((entry.name, "missing"))
            elif sha256_file(file) != entry.sha256:
                out.append((entry.name, "MISMATCH"))
            else:
                out.append((entry.name, "ok"))
        return out

    def fetch(self, name: str, opener: Callable[[str], Any] = urllib.request.urlopen) -> Path:
        """Download an entry's URL to a temporary file, verify the checksum, then install it."""
        entry = self.get(name)
        if not entry.url:
            raise ModelIntegrityError(f"model {name} has no download URL")
        target = self.file_of(entry)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(delete=False, dir=target.parent, suffix=".part") as tmp:
            tmp_path = Path(tmp.name)
            with opener(entry.url) as response:
                shutil.copyfileobj(response, tmp)
        try:
            actual = sha256_file(tmp_path)
            if actual != entry.sha256:
                raise ModelIntegrityError(f"download of {name} has checksum {actual[:12]}..., expected "
                                          f"{entry.sha256[:12]}...; not installed")
            tmp_path.replace(target)
        finally:
            tmp_path.unlink(missing_ok=True)
        return target


def integrity_fault(err: Exception) -> SystemHealth:
    """The SYSTEM_FAULT reported when a model fails its integrity check."""
    return SystemHealth(component=HealthComponent.SYSTEM, status=Status.ERROR,
                        detail=f"model integrity check failed: {err}", t_utc=utc_now())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m lifeguard.common.models")
    parser.add_argument("command", choices=["verify", "hash", "fetch"])
    parser.add_argument("target", nargs="?", help="file (hash) or model name (fetch)")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    args = parser.parse_args(argv)
    if args.command == "hash":
        if not args.target:
            parser.error("hash needs a file")
        print(sha256_file(args.target))
        return 0
    try:
        registry = ModelRegistry.load(args.registry)
        if args.command == "fetch":
            if not args.target:
                parser.error("fetch needs a model name")
            print(f"installed {registry.fetch(args.target)}")
            return 0
    except (ConfigError, ModelIntegrityError, ValidationError, OSError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 3
    failed = 0
    for name, status in registry.report():
        print(f"{status:9s} {name}")
        failed += status == "MISMATCH"
    return 3 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
