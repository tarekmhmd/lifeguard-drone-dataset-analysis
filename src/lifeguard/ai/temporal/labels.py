"""Behaviour labels (M1-04): taxonomy, interval CSV, annotator agreement, and training samples.

The human-readable rules are in ``docs/ai/annotation_guide.md``; ``label_taxonomy.json`` (next to this
module) is the machine-readable version used here. Interval CSV columns follow M1-04 exactly:
``session_id,video_file,track_id,t_start,t_end,label,annotator,notes`` (times in ``t_media`` seconds).

CLI::

    python -m lifeguard.ai.temporal.labels validate intervals.csv
    python -m lifeguard.ai.temporal.labels agreement annotator_a.csv annotator_b.csv [--step 0.5]
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from lifeguard.ai.temporal.dataset import BEHAVIOR_CLASSES, Sample, vectorize
from lifeguard.contracts.enums import Status
from lifeguard.contracts.errors import ContractValidationError
from lifeguard.contracts.types import FrameResult

TAXONOMY_PATH = Path(__file__).with_name("label_taxonomy.json")
INTERVAL_COLUMNS: tuple[str, ...] = (
    "session_id", "video_file", "track_id", "t_start", "t_end", "label", "annotator", "notes",
)
_EPS = 1e-9


class LabelDef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    contract_class: str | None
    min_duration_s: float = Field(gt=0)
    summary: str
    criteria: list[str]
    decision_rule: str | None = None
    not_this_label_when: list[str]


class Taxonomy(BaseModel):
    model_config = ConfigDict(extra="allow")

    taxonomy_version: str
    principles: list[str]
    labels: list[LabelDef]

    @model_validator(mode="after")
    def _classes_exist(self) -> Taxonomy:
        for d in self.labels:
            if d.contract_class is not None and d.contract_class not in BEHAVIOR_CLASSES:
                raise ValueError(f"{d.label}: unknown contract class {d.contract_class}")
        return self

    def get(self, label: str) -> LabelDef:
        for d in self.labels:
            if d.label == label:
                return d
        raise KeyError(label)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(d.label for d in self.labels)


@cache
def load_taxonomy(path: Path = TAXONOMY_PATH) -> Taxonomy:
    return Taxonomy.model_validate(json.loads(path.read_text(encoding="utf-8")))


class BehaviorInterval(BaseModel):
    """One labelled interval of one track."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: str = Field(pattern=r"^[A-Za-z0-9_.\-]{1,64}$")
    video_file: str = ""
    track_id: int = Field(ge=1)
    t_start: float = Field(ge=0)
    t_end: float
    label: str
    annotator: str = Field(min_length=1, max_length=64)
    notes: str = ""

    @model_validator(mode="after")
    def _valid(self) -> BehaviorInterval:
        if not self.t_end > self.t_start:
            raise ValueError("t_end must be greater than t_start")
        try:
            definition = load_taxonomy().get(self.label)
        except KeyError as err:
            raise ValueError(f"unknown label {self.label!r}; allowed: {', '.join(load_taxonomy().names)}") from err
        if self.t_end - self.t_start + _EPS < definition.min_duration_s:
            raise ValueError(f"'{self.label}' intervals must last >= {definition.min_duration_s} s")
        return self

    def covers(self, t: float) -> bool:
        return self.t_start <= t <= self.t_end


def read_intervals(path: Path | str) -> list[BehaviorInterval]:
    """Read and validate an interval CSV (header line 1). Overlaps within one annotator's track are errors."""
    source = str(path)
    with Path(path).open(encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows or tuple(rows[0]) != INTERVAL_COLUMNS:
        raise ContractValidationError(f"header must be {','.join(INTERVAL_COLUMNS)}", source=source, line=1)
    out: list[BehaviorInterval] = []
    for n, row in enumerate(rows[1:], start=2):
        if not any(cell.strip() for cell in row):
            continue
        if len(row) != len(INTERVAL_COLUMNS):
            raise ContractValidationError("wrong number of columns", source=source, line=n)
        try:
            out.append(BehaviorInterval(**dict(zip(INTERVAL_COLUMNS, row, strict=True))))
        except ValidationError as err:
            raise ContractValidationError(str(err), source=source, line=n) from err
    by_key: dict[tuple[str, int, str], list[BehaviorInterval]] = {}
    for iv in out:
        by_key.setdefault((iv.session_id, iv.track_id, iv.annotator), []).append(iv)
    for key, group in by_key.items():
        group.sort(key=lambda iv: iv.t_start)
        for a, b in itertools.pairwise(group):
            if b.t_start < a.t_end - _EPS:
                raise ContractValidationError(f"overlapping intervals for {key}: {a.t_start}-{a.t_end} and "
                                              f"{b.t_start}-{b.t_end}", source=source)
    return out


def write_intervals(path: Path | str, intervals: Iterable[BehaviorInterval]) -> int:
    rows = 0
    with Path(path).open("w", encoding="utf-8", newline="\n") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(INTERVAL_COLUMNS)
        for iv in intervals:
            writer.writerow([iv.session_id, iv.video_file, iv.track_id, repr(iv.t_start), repr(iv.t_end),
                             iv.label, iv.annotator, iv.notes])
            rows += 1
    return rows


@dataclass(frozen=True)
class AgreementReport:
    """Agreement between two annotators on the time both of them labelled."""

    n_points: int
    percent: float | None
    kappa: float | None
    confusion: dict[tuple[str, str], int]

    def line(self) -> str:
        def fmt(v: float | None) -> str:
            return "na" if v is None else f"{v:.3f}"

        return f"points={self.n_points} agreement={fmt(self.percent)} kappa={fmt(self.kappa)}"


def _label_at(intervals: list[BehaviorInterval], t: float) -> str | None:
    for iv in intervals:
        if iv.covers(t):
            return iv.label
    return None


def agreement(a: Iterable[BehaviorInterval], b: Iterable[BehaviorInterval], step_s: float = 0.5) -> AgreementReport:
    """Cohen's kappa and percent agreement over a time grid, per (session, track) labelled by both."""
    group_a: dict[tuple[str, int], list[BehaviorInterval]] = {}
    group_b: dict[tuple[str, int], list[BehaviorInterval]] = {}
    for iv in a:
        group_a.setdefault((iv.session_id, iv.track_id), []).append(iv)
    for iv in b:
        group_b.setdefault((iv.session_id, iv.track_id), []).append(iv)
    pairs: list[tuple[str, str]] = []
    for key in sorted(group_a.keys() & group_b.keys()):
        ia, ib = group_a[key], group_b[key]
        start = max(min(iv.t_start for iv in ia), min(iv.t_start for iv in ib))
        end = min(max(iv.t_end for iv in ia), max(iv.t_end for iv in ib))
        for k in range(math.floor((end - start) / step_s + _EPS) + 1 if end >= start else 0):
            t = start + k * step_s
            la, lb = _label_at(ia, t), _label_at(ib, t)
            if la is not None and lb is not None:
                pairs.append((la, lb))
    if not pairs:
        return AgreementReport(0, None, None, {})
    n = len(pairs)
    observed = sum(1 for x, y in pairs if x == y) / n
    count_a = Counter(x for x, _ in pairs)
    count_b = Counter(y for _, y in pairs)
    expected = sum(count_a[k] * count_b[k] for k in count_a) / (n * n)
    kappa = 1.0 if expected >= 1.0 else (observed - expected) / (1.0 - expected)
    return AgreementReport(n, observed, kappa, dict(Counter(pairs)))


def build_interval_samples(
    frames: Iterable[FrameResult],
    intervals: Iterable[BehaviorInterval],
    *,
    stride_s: float = 0.5,
) -> list[Sample]:
    """Training samples from labelled intervals. Only labelled, non-'uncertain' time is used; unlabelled
    time never becomes a negative example. ``track_id`` refers to the system's own track IDs."""
    taxonomy = load_taxonomy()
    by_track: dict[tuple[str, int], list[BehaviorInterval]] = {}
    for iv in intervals:
        by_track.setdefault((iv.session_id, iv.track_id), []).append(iv)
    last_emit: dict[str, float] = {}
    samples: list[Sample] = []
    ordered = sorted((f for f in frames if f.status != Status.ERROR and f.t_media is not None),
                     key=lambda f: f.t_media)  # type: ignore[arg-type, return-value]
    for frame in ordered:
        t = float(frame.t_media)  # type: ignore[arg-type]
        for track in frame.tracks:
            if not track.in_water or not track.features:
                continue
            label = _label_at(by_track.get((frame.session_id, track.track_id), []), t)
            if label is None:
                continue
            contract_class = taxonomy.get(label).contract_class
            if contract_class is None:
                continue  # 'uncertain'
            if t - last_emit.get(track.track_uid, -math.inf) < stride_s - _EPS:
                continue
            last_emit[track.track_uid] = t
            samples.append(Sample(frame.session_id, track.track_uid, t, vectorize(track.features), contract_class))
    return samples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m lifeguard.ai.temporal.labels")
    sub = parser.add_subparsers(dest="command", required=True)
    v = sub.add_parser("validate")
    v.add_argument("file", type=Path)
    ag = sub.add_parser("agreement")
    ag.add_argument("a", type=Path)
    ag.add_argument("b", type=Path)
    ag.add_argument("--step", type=float, default=0.5)
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            intervals = read_intervals(args.file)
            counts = Counter(iv.label for iv in intervals)
            print(f"valid intervals={len(intervals)} " + " ".join(f"{k}={counts[k]}" for k in sorted(counts)))
            return 0
        print(agreement(read_intervals(args.a), read_intervals(args.b), args.step).line())
        return 0
    except (ContractValidationError, OSError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
