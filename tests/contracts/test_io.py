"""TP-42..TP-45: JSONL and events CSV I/O."""
from __future__ import annotations

from pathlib import Path

import pytest

from lifeguard.contracts.errors import ContractValidationError
from lifeguard.contracts.io import read_events_csv, read_jsonl, write_events_csv, write_jsonl
from lifeguard.contracts.types import FrameResult, GroundTruthEvent
from tests.contracts.conftest import make_frame, make_ground_truth


def test_jsonl_round_trip(tmp_path: Path) -> None:
    """write_jsonl returns the count and read_jsonl round-trips the items."""
    frames = [FrameResult(**make_frame(frame_idx=i)) for i in range(5)]
    path = tmp_path / "frames.jsonl"

    assert write_jsonl(path, frames) == 5
    assert read_jsonl(path, FrameResult) == frames


def test_jsonl_bad_line_reports_line(tmp_path: Path) -> None:
    """An invalid line reports its 1-based line number and the source path."""
    path = tmp_path / "bad.jsonl"
    good = FrameResult(**make_frame()).model_dump_json()
    path.write_text(f"{good}\n{good}\n{{not json}}\n", encoding="utf-8", newline="\n")

    with pytest.raises(ContractValidationError) as excinfo:
        read_jsonl(path, FrameResult)

    assert excinfo.value.line == 3
    assert excinfo.value.source == str(path)


def test_newlines_are_lf(tmp_path: Path) -> None:
    """JSONL is written with LF, never CRLF."""
    path = tmp_path / "lf.jsonl"
    write_jsonl(path, [FrameResult(**make_frame())])

    raw = path.read_bytes()
    assert b"\n" in raw
    assert b"\r\n" not in raw


def test_events_csv_round_trip_and_header(tmp_path: Path) -> None:
    """Events round-trip through CSV and a wrong header reports line 1."""
    first = GroundTruthEvent(**make_ground_truth())
    second = GroundTruthEvent(
        **make_ground_truth(
            event_id="evt_2",
            t_end=40.0,
            track_uid=None,
            pool_x=None,
            pool_y=None,
        )
    )
    path = tmp_path / "events.csv"

    assert write_events_csv(path, [first, second]) == 2
    assert read_events_csv(path) == [first, second]

    bad = tmp_path / "bad.csv"
    bad.write_text("event_id,session_id,wrong\n", encoding="utf-8", newline="\n")
    with pytest.raises(ContractValidationError) as excinfo:
        read_events_csv(bad)
    assert excinfo.value.line == 1
