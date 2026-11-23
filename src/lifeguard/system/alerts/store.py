"""SQLite incident log (Blueprint M9: "incident record", NFR-4 traceability). Standard library only.

Every RiskEvent, acknowledgement and DispatchCommand is stored as its contract JSON next to a few indexed
columns, so an incident can be audited and replayed later. Dismissals are queryable as hard negatives.
Thread-safe: one shared connection guarded by a lock (the dashboard and the frame loop use it together).
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any

from lifeguard.contracts.types import Acknowledgement, DispatchCommand, RiskEvent

if TYPE_CHECKING:
    from typing_extensions import Self

_SCHEMA = """
CREATE TABLE IF NOT EXISTS incidents (
    incident_id   TEXT PRIMARY KEY,
    cam_id        TEXT NOT NULL,
    session_id    TEXT NOT NULL,
    track_uid     TEXT NOT NULL,
    opened_t_utc  TEXT NOT NULL,
    closed_t_utc  TEXT,
    level         TEXT NOT NULL,
    max_level     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS risk_events (
    event_id     TEXT PRIMARY KEY,
    incident_id  TEXT NOT NULL REFERENCES incidents(incident_id),
    to_level     TEXT NOT NULL,
    reason       TEXT NOT NULL,
    t_utc        TEXT NOT NULL,
    json         TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS acknowledgements (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    incident_id  TEXT NOT NULL REFERENCES incidents(incident_id),
    event_id     TEXT NOT NULL,
    decision     TEXT NOT NULL,
    ack_by       TEXT NOT NULL,
    level_before TEXT NOT NULL,
    t_utc        TEXT NOT NULL,
    json         TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS clips (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    incident_id  TEXT NOT NULL,
    path         TEXT NOT NULL,
    t_utc        TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS dispatch_commands (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    incident_id  TEXT NOT NULL REFERENCES incidents(incident_id),
    json         TEXT NOT NULL
);
"""


class IncidentStore:
    """Thin wrapper around one SQLite database (``":memory:"`` by default)."""

    def __init__(self, path: Path | str = ":memory:") -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def _write(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        with self._lock:
            self._conn.execute(sql, params)
            self._conn.commit()

    def _read(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    # ------------------------------------------------------------------ writes

    def open_incident(self, incident_id: str, event: RiskEvent) -> None:
        self._write(
            "INSERT INTO incidents (incident_id, cam_id, session_id, track_uid, opened_t_utc, level, max_level) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (incident_id, event.cam_id, event.session_id, event.track_uid, event.t_utc,
             event.to_level.value, event.to_level.value),
        )

    def update_incident(
        self, incident_id: str, *, track_uid: str, level: str, max_level: str, closed_t_utc: str | None
    ) -> None:
        self._write(
            "UPDATE incidents SET track_uid = ?, level = ?, max_level = ?, closed_t_utc = ? "
            "WHERE incident_id = ?",
            (track_uid, level, max_level, closed_t_utc, incident_id),
        )

    def add_event(self, incident_id: str, event: RiskEvent) -> None:
        self._write(
            "INSERT OR IGNORE INTO risk_events (event_id, incident_id, to_level, reason, t_utc, json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (event.event_id, incident_id, event.to_level.value, event.reason.value, event.t_utc,
             event.model_dump_json()),
        )

    def add_ack(self, incident_id: str, ack: Acknowledgement, level_before: str) -> None:
        self._write(
            "INSERT INTO acknowledgements (incident_id, event_id, decision, ack_by, level_before, t_utc, json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (incident_id, ack.event_id, ack.decision.value, ack.by, level_before, ack.t_utc,
             ack.model_dump_json()),
        )

    def add_dispatch(self, incident_id: str, command: DispatchCommand) -> None:
        self._write(
            "INSERT INTO dispatch_commands (incident_id, json) VALUES (?, ?)",
            (incident_id, command.model_dump_json()),
        )

    def add_clip(self, incident_id: str, path: str, t_utc: str) -> None:
        self._write(
            "INSERT INTO clips (incident_id, path, t_utc) VALUES (?, ?, ?)", (incident_id, path, t_utc)
        )

    # ------------------------------------------------------------------- reads

    def clips(self, incident_id: str | None = None) -> list[dict[str, Any]]:
        if incident_id is None:
            rows = self._read("SELECT * FROM clips ORDER BY id")
        else:
            rows = self._read(
                "SELECT * FROM clips WHERE incident_id = ? ORDER BY id", (incident_id,)
            )
        return [dict(r) for r in rows]

    def incidents(self) -> list[dict[str, Any]]:
        rows = self._read("SELECT * FROM incidents ORDER BY opened_t_utc, incident_id")
        return [dict(r) for r in rows]

    def events(self, incident_id: str) -> list[RiskEvent]:
        rows = self._read(
            "SELECT json FROM risk_events WHERE incident_id = ? ORDER BY rowid", (incident_id,)
        )
        return [RiskEvent.model_validate_json(r["json"]) for r in rows]

    def dispatches(self) -> list[DispatchCommand]:
        rows = self._read("SELECT json FROM dispatch_commands ORDER BY id")
        return [DispatchCommand.model_validate_json(r["json"]) for r in rows]

    def hard_negatives(self) -> list[dict[str, Any]]:
        """Dismissed incidents: lifeguard said "not an emergency" (training material, Blueprint M8)."""
        rows = self._read(
            "SELECT a.incident_id, a.event_id, a.ack_by, a.level_before, a.t_utc, i.track_uid, i.session_id "
            "FROM acknowledgements a JOIN incidents i ON i.incident_id = a.incident_id "
            "WHERE a.decision = 'dismiss' ORDER BY a.id"
        )
        return [dict(r) for r in rows]
