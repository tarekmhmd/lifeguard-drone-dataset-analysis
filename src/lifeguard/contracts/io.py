"""JSONL and CSV serialisation helpers for contract models."""
from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path
from typing import TypeVar

from pydantic import ValidationError

from lifeguard.contracts.enums import GroundTruthType
from lifeguard.contracts.errors import ContractValidationError
from lifeguard.contracts.types import ContractModel, GroundTruthEvent

ModelT = TypeVar("ModelT", bound=ContractModel)

_EVENTS_CSV_HEADER = [
    "event_id",
    "session_id",
    "video_file",
    "type",
    "t_onset",
    "t_end",
    "track_uid",
    "pool_x",
    "pool_y",
]


def write_jsonl(path: Path | str, items: Iterable[ContractModel]) -> int:
    """Write one ``model_dump_json()`` per line as UTF-8 with LF newlines.

    Args:
        path: Destination file. Parent directories are created when needed.
        items: Contract models to serialise, one per line.

    Returns:
        The number of lines written. An existing file is overwritten.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for item in items:
            file.write(item.model_dump_json())
            file.write("\n")
            count += 1
    return count


def read_jsonl(path: Path | str, model: type[ModelT]) -> list[ModelT]:
    """Parse each non-empty JSONL line with ``model.model_validate_json``.

    Args:
        path: Source file.
        model: Contract model used to validate each line.

    Returns:
        The list of validated models.

    Raises:
        FileNotFoundError: If the file does not exist.
        ContractValidationError: On the first invalid line, with ``source`` set
            to the path and ``line`` set to the 1-based line number.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")

    items: list[ModelT] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            items.append(model.model_validate_json(line))
        except ValidationError as err:
            raise ContractValidationError(
                f"invalid {model.__name__} entry",
                source=str(path),
                line=line_number,
            ) from err
    return items


def write_events_csv(path: Path | str, events: Iterable[GroundTruthEvent]) -> int:
    """Write ground-truth events as CSV with the exact column order of SPEC 4.6.10.

    Args:
        path: Destination file. Parent directories are created when needed.
        events: Ground-truth events to write after the header row.

    Returns:
        The number of event rows written. An existing file is overwritten.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(_EVENTS_CSV_HEADER)
        for event in events:
            writer.writerow(
                [
                    event.event_id,
                    event.session_id,
                    event.video_file,
                    event.type.value,
                    repr(event.t_onset),
                    "" if event.t_end is None else repr(event.t_end),
                    "" if event.track_uid is None else event.track_uid,
                    "" if event.pool_x is None else repr(event.pool_x),
                    "" if event.pool_y is None else repr(event.pool_y),
                ]
            )
            count += 1
    return count


def read_events_csv(path: Path | str) -> list[GroundTruthEvent]:
    """Read a ground-truth events CSV produced by :func:`write_events_csv`.

    Args:
        path: Source file.

    Returns:
        The list of ground-truth events (the header row is not included).

    Raises:
        ContractValidationError: If the header is wrong (line 1) or a row is
            invalid (its 1-based line number), with ``source`` set to the path.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.reader(file))

    if not rows or rows[0] != _EVENTS_CSV_HEADER:
        raise ContractValidationError("wrong events CSV header", source=str(path), line=1)

    events: list[GroundTruthEvent] = []
    for line_number, row in enumerate(rows[1:], start=2):
        if len(row) != len(_EVENTS_CSV_HEADER):
            raise ContractValidationError(
                "wrong number of columns",
                source=str(path),
                line=line_number,
            )
        try:
            events.append(_event_from_row(row))
        except (ValueError, ValidationError) as err:
            raise ContractValidationError(
                "invalid events CSV row",
                source=str(path),
                line=line_number,
            ) from err
    return events


def _parse_optional_float(value: str) -> float | None:
    """Return None for an empty cell, otherwise the parsed float."""
    if value == "":
        return None
    return float(value)


def _event_from_row(row: list[str]) -> GroundTruthEvent:
    """Build a GroundTruthEvent from a CSV row (without the header)."""
    return GroundTruthEvent(
        event_id=row[0],
        session_id=row[1],
        video_file=row[2],
        type=GroundTruthType(row[3]),
        t_onset=float(row[4]),
        t_end=_parse_optional_float(row[5]),
        track_uid=row[6] or None,
        pool_x=_parse_optional_float(row[7]),
        pool_y=_parse_optional_float(row[8]),
    )
