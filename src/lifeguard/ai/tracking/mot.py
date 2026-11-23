"""MOTChallenge export of tracker output, for tracking evaluation with TrackEval (M1-05).

Each line: ``frame,id,bb_left,bb_top,bb_width,bb_height,conf,x,y,z`` with 1-based frame numbers, pixel
boxes and ``-1`` for the unused world coordinates. By default only visible (``tracked``) boxes are
exported; ``lost`` boxes repeat the last position and would be scored as false positives.

TrackEval layout written by ``export_for_trackeval``::

    <root>/gt/mot_challenge/<BENCH>-<SPLIT>/<seq>/seqinfo.ini        (gt/gt.txt comes from CVAT, MOT 1.1)
    <root>/trackers/mot_challenge/<BENCH>-<SPLIT>/<tracker>/data/<seq>.txt

CLI::

    python -m lifeguard.ai.tracking.mot --frames run_frames.jsonl --root eval --seq pool_clip01 [--fps 10]

The metrics themselves (IDF1, HOTA, ID switches) need annotated pool clips and the TrackEval package; this
module only produces correct input files.
"""
from __future__ import annotations

import argparse
import configparser
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from lifeguard.contracts.enums import TrackState
from lifeguard.contracts.errors import ContractValidationError
from lifeguard.contracts.io import read_jsonl
from lifeguard.contracts.types import FrameResult

N_COLUMNS = 10


@dataclass(frozen=True)
class MotRow:
    frame: int
    track_id: int
    left: float
    top: float
    width: float
    height: float
    conf: float

    def line(self) -> str:
        return (f"{self.frame},{self.track_id},{self.left:.2f},{self.top:.2f},{self.width:.2f},"
                f"{self.height:.2f},{self.conf:.3f},-1,-1,-1")


def frames_to_rows(frames: Iterable[FrameResult], include_lost: bool = False) -> list[MotRow]:
    rows: list[MotRow] = []
    for f in frames:
        for t in f.tracks:
            if t.state == TrackState.LOST and not include_lost:
                continue
            x1, y1, x2, y2 = t.box_xyxy
            rows.append(MotRow(f.frame_idx + 1, t.track_id, x1, y1, x2 - x1, y2 - y1, t.det_conf))
    return sorted(rows, key=lambda r: (r.frame, r.track_id))


def write_mot(path: Path | str, frames: Iterable[FrameResult], include_lost: bool = False) -> int:
    rows = frames_to_rows(frames, include_lost)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(r.line() + "\n" for r in rows), encoding="utf-8", newline="\n")
    return len(rows)


def read_mot(path: Path | str) -> list[MotRow]:
    """Parse and validate a MOT file. Raises ContractValidationError(source, line)."""
    source = str(path)
    rows: list[MotRow] = []
    seen: set[tuple[int, int]] = set()
    for n, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        cells = raw.split(",")
        if len(cells) != N_COLUMNS:
            raise ContractValidationError(f"expected {N_COLUMNS} columns, got {len(cells)}", source=source, line=n)
        try:
            frame, track_id = int(cells[0]), int(float(cells[1]))
            left, top, width, height, conf = (float(c) for c in cells[2:7])
        except ValueError as err:
            raise ContractValidationError(f"not a number: {err}", source=source, line=n) from err
        if frame < 1 or track_id < 1:
            raise ContractValidationError("frame and id must be >= 1", source=source, line=n)
        if width <= 0 or height <= 0:
            raise ContractValidationError("box width and height must be > 0", source=source, line=n)
        if (frame, track_id) in seen:
            raise ContractValidationError(f"duplicate id {track_id} in frame {frame}", source=source, line=n)
        seen.add((frame, track_id))
        rows.append(MotRow(frame, track_id, left, top, width, height, conf))
    return rows


def write_seqinfo(path: Path | str, name: str, frames: list[FrameResult], fps: float) -> None:
    if not frames:
        raise ValueError("no frames to describe")
    info = configparser.ConfigParser()
    info.optionxform = str  # keep the MOT key capitalisation
    info["Sequence"] = {
        "name": name,
        "imDir": "img1",
        "frameRate": f"{fps:g}",
        "seqLength": str(max(f.frame_idx for f in frames) + 1),
        "imWidth": str(frames[0].frame_width),
        "imHeight": str(frames[0].frame_height),
        "imExt": ".jpg",
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as fh:
        info.write(fh, space_around_delimiters=False)


def export_for_trackeval(
    frames: list[FrameResult],
    root: Path | str,
    seq: str,
    fps: float,
    tracker: str = "lifeguard",
    benchmark: str = "POOL",
    split: str = "test",
    include_lost: bool = False,
) -> tuple[Path, Path]:
    """Write the tracker file and seqinfo.ini in TrackEval's MOTChallenge layout; returns both paths."""
    bench = f"{benchmark}-{split}"
    tracker_file = Path(root) / "trackers" / "mot_challenge" / bench / tracker / "data" / f"{seq}.txt"
    seqinfo = Path(root) / "gt" / "mot_challenge" / bench / seq / "seqinfo.ini"
    write_mot(tracker_file, frames, include_lost)
    write_seqinfo(seqinfo, seq, frames, fps)
    return tracker_file, seqinfo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m lifeguard.ai.tracking.mot")
    parser.add_argument("--frames", type=Path, required=True, help="FrameResult JSONL (real engine output)")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seq", required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--tracker", default="lifeguard")
    parser.add_argument("--include-lost", action="store_true")
    args = parser.parse_args(argv)
    try:
        frames = read_jsonl(args.frames, FrameResult)
        tracker_file, seqinfo = export_for_trackeval(frames, args.root, args.seq, args.fps, args.tracker,
                                                     include_lost=args.include_lost)
    except (OSError, ContractValidationError, ValueError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    print(f"rows={len(read_mot(tracker_file))} tracker_file={tracker_file} seqinfo={seqinfo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
