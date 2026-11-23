"""MOT export for tracking evaluation: format, layout, validation and CLI."""
from __future__ import annotations

import configparser
import subprocess
import sys
from functools import cache
from pathlib import Path

import numpy as np
import pytest

from lifeguard.ai.detection.base import Detection, Detector
from lifeguard.ai.perception.real_engine import RealPerceptionEngine
from lifeguard.ai.sim.scenarios import make_scenario
from lifeguard.ai.tracking.mot import (
    N_COLUMNS,
    export_for_trackeval,
    frames_to_rows,
    read_mot,
    write_mot,
)
from lifeguard.contracts.enums import TrackState
from lifeguard.contracts.errors import ContractValidationError
from lifeguard.contracts.io import write_jsonl
from lifeguard.contracts.types import FrameResult
from lifeguard.system.geometry.stub import StubPoolGeometry

ROOT = Path(__file__).resolve().parents[2]


class _Replay(Detector):
    def __init__(self, frames: tuple[FrameResult, ...]) -> None:
        self._frames, self._i = frames, 0

    @property
    def model_version(self) -> str:
        return "synthetic_boxes"

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        f = self._frames[self._i]
        self._i += 1
        return [Detection(t.box_xyxy, t.det_conf, t.cls) for t in f.tracks if t.state == TrackState.TRACKED]


@cache
def _frames(kind: str = "dive") -> tuple[FrameResult, ...]:
    scenario = make_scenario(kind, seed=0)
    engine = RealPerceptionEngine(_Replay(scenario.frames), StubPoolGeometry(), "cam_sim", "s1")
    image = np.zeros((1080, 1920, 3), np.uint8)
    return tuple(engine.process(image, f.frame_idx, f.t_capture, f.t_utc, f.t_media) for f in scenario.frames)


def test_rows_are_one_based_visible_only_and_inside_the_image(tmp_path: Path) -> None:
    frames = list(_frames())
    path = tmp_path / "seq.txt"
    n = write_mot(path, frames)
    rows = read_mot(path)
    assert len(rows) == n > 0
    visible = sum(1 for f in frames for t in f.tracks if t.state == TrackState.TRACKED)
    assert n == visible
    first_visible = min(f.frame_idx for f in frames if any(t.state == TrackState.TRACKED for t in f.tracks))
    assert min(r.frame for r in rows) == first_visible + 1  # MOT frames are 1-based
    for r in rows:
        assert r.left >= 0 and r.top >= 0 and r.left + r.width <= 1920 + 1e-6 and r.top + r.height <= 1080 + 1e-6
    line = path.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert len(line) == N_COLUMNS and line[-3:] == ["-1", "-1", "-1"]
    with_lost = frames_to_rows(frames, include_lost=True)
    assert len(with_lost) > n  # the dive leaves lost boxes behind


def test_trackeval_layout_and_seqinfo(tmp_path: Path) -> None:
    frames = list(_frames())
    tracker_file, seqinfo = export_for_trackeval(frames, tmp_path, "clip01", fps=10.0)
    assert tracker_file == tmp_path / "trackers" / "mot_challenge" / "POOL-test" / "lifeguard" / "data" / "clip01.txt"
    assert seqinfo == tmp_path / "gt" / "mot_challenge" / "POOL-test" / "clip01" / "seqinfo.ini"
    info = configparser.ConfigParser()
    info.optionxform = str
    info.read(seqinfo, encoding="utf-8")
    seq = info["Sequence"]
    assert seq["name"] == "clip01" and seq["frameRate"] == "10"
    assert int(seq["seqLength"]) == len(frames)
    assert (seq["imWidth"], seq["imHeight"]) == ("1920", "1080")


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("1,1,0,0,10,10,0.9,-1,-1\n", "columns"),
        ("0,1,0,0,10,10,0.9,-1,-1,-1\n", ">= 1"),
        ("1,1,0,0,0,10,0.9,-1,-1,-1\n", "width"),
        ("1,1,0,0,10,10,0.9,-1,-1,-1\n1,1,5,5,10,10,0.8,-1,-1,-1\n", "duplicate"),
        ("1,a,0,0,10,10,0.9,-1,-1,-1\n", "number"),
    ],
)
def test_invalid_mot_files(tmp_path: Path, content: str, message: str) -> None:
    path = tmp_path / "bad.txt"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ContractValidationError, match=message):
        read_mot(path)


def test_mot_cli(tmp_path: Path) -> None:
    frames_path = tmp_path / "frames.jsonl"
    write_jsonl(frames_path, _frames("normal_swim"))
    proc = subprocess.run([sys.executable, "-m", "lifeguard.ai.tracking.mot", "--frames", str(frames_path),
                           "--root", str(tmp_path / "eval"), "--seq", "clip02"],
                          cwd=ROOT, capture_output=True, text=True, timeout=120, check=False)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.startswith("rows=")
    missing = subprocess.run([sys.executable, "-m", "lifeguard.ai.tracking.mot", "--frames",
                              str(tmp_path / "absent.jsonl"), "--root", str(tmp_path), "--seq", "x"],
                             cwd=ROOT, capture_output=True, text=True, timeout=60, check=False)
    assert missing.returncode == 2
