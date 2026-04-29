"""SQLite event log — plan §3.3 schema (raw SQL, no SQLAlchemy).

Three tables:

    runs(id, started_at, ended_at, script_path, git_branch, status)
    events(id, ts, run_id, kind, payload_json)              -- append-only
    decisions(id, run_id, ts, fix_template, diff_text, approved, applied_at)

Default path: ``~/.lensml/runs/{run_id}/events.db``. Tests use ``:memory:``.

Synchronous SQLite is intentional for v1 — write volume is bounded by the
plan §3.1 tick rate (10-50 Hz). Each insert is sub-millisecond on modern
SSD; safe inside the FastAPI async event loop. If volume grows, swap to
aiosqlite or a worker thread (v0.2).
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id           TEXT PRIMARY KEY,
    started_at   REAL NOT NULL,
    ended_at     REAL,
    script_path  TEXT,
    git_branch   TEXT,
    status       TEXT NOT NULL DEFAULT 'running'
);

CREATE TABLE IF NOT EXISTS events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            REAL NOT NULL,
    run_id        TEXT NOT NULL,
    kind          TEXT NOT NULL,
    payload_json  TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(id)
);

CREATE INDEX IF NOT EXISTS idx_events_run_ts ON events(run_id, ts);
CREATE INDEX IF NOT EXISTS idx_events_run_id_id ON events(run_id, id);

CREATE TABLE IF NOT EXISTS decisions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT NOT NULL,
    ts            REAL NOT NULL,
    fix_template  TEXT NOT NULL,
    diff_text     TEXT,
    approved      INTEGER,
    applied_at    REAL,
    FOREIGN KEY (run_id) REFERENCES runs(id)
);
"""


class EventStore:
    """SQLite-backed event log with the plan §3.3 schema.

    Connection is opened with ``check_same_thread=False`` so the FastAPI
    async event loop can write from the bridge task and read from
    request handlers without thread-affinity errors.
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            isolation_level=None,  # autocommit
        )
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.executescript(SCHEMA)

    # ---- runs ----

    def begin_run(
        self,
        run_id: str,
        script_path: str | None = None,
        git_branch: str | None = None,
    ) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO runs (id, started_at, script_path, git_branch, status) "
            "VALUES (?, ?, ?, ?, 'running')",
            (run_id, time.time(), script_path, git_branch),
        )

    def end_run(self, run_id: str, status: str = "completed") -> None:
        self.conn.execute(
            "UPDATE runs SET ended_at = ?, status = ? WHERE id = ?",
            (time.time(), status, run_id),
        )

    def list_runs(self) -> list[dict[str, Any]]:
        cur = self.conn.execute(
            "SELECT id, started_at, ended_at, script_path, git_branch, status "
            "FROM runs ORDER BY started_at DESC"
        )
        cols = ("id", "started_at", "ended_at", "script_path", "git_branch", "status")
        return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        cur = self.conn.execute(
            "SELECT id, started_at, ended_at, script_path, git_branch, status "
            "FROM runs WHERE id = ?",
            (run_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        cols = ("id", "started_at", "ended_at", "script_path", "git_branch", "status")
        return dict(zip(cols, row, strict=True))

    # ---- events (append-only) ----

    def append_event(
        self,
        run_id: str,
        kind: str,
        payload: dict[str, Any],
        ts: float | None = None,
    ) -> int:
        ts = ts if ts is not None else time.time()
        cur = self.conn.execute(
            "INSERT INTO events (ts, run_id, kind, payload_json) VALUES (?, ?, ?, ?)",
            (ts, run_id, kind, json.dumps(payload, default=str)),
        )
        return int(cur.lastrowid or 0)

    def get_events(
        self,
        run_id: str,
        *,
        since_id: int = 0,
        limit: int = 1000,
        kind: str | None = None,
    ) -> list[dict[str, Any]]:
        if kind is None:
            cur = self.conn.execute(
                "SELECT id, ts, kind, payload_json FROM events "
                "WHERE run_id = ? AND id > ? ORDER BY id ASC LIMIT ?",
                (run_id, since_id, limit),
            )
        else:
            cur = self.conn.execute(
                "SELECT id, ts, kind, payload_json FROM events "
                "WHERE run_id = ? AND kind = ? AND id > ? ORDER BY id ASC LIMIT ?",
                (run_id, kind, since_id, limit),
            )
        return [
            {
                "id": row[0],
                "ts": row[1],
                "kind": row[2],
                "payload": json.loads(row[3]),
            }
            for row in cur.fetchall()
        ]

    def event_count(self, run_id: str) -> int:
        cur = self.conn.execute("SELECT COUNT(*) FROM events WHERE run_id = ?", (run_id,))
        return int(cur.fetchone()[0])

    # ---- decisions ----

    def record_decision(
        self,
        run_id: str,
        fix_template: str,
        diff_text: str | None = None,
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO decisions (run_id, ts, fix_template, diff_text) VALUES (?, ?, ?, ?)",
            (run_id, time.time(), fix_template, diff_text),
        )
        return int(cur.lastrowid or 0)

    def resolve_decision(self, decision_id: int, *, approved: bool) -> None:
        self.conn.execute(
            "UPDATE decisions SET approved = ?, applied_at = ? WHERE id = ?",
            (1 if approved else 0, time.time() if approved else None, decision_id),
        )

    def list_decisions(self, run_id: str) -> list[dict[str, Any]]:
        cur = self.conn.execute(
            "SELECT id, ts, fix_template, diff_text, approved, applied_at "
            "FROM decisions WHERE run_id = ? ORDER BY ts ASC",
            (run_id,),
        )
        cols = ("id", "ts", "fix_template", "diff_text", "approved", "applied_at")
        return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]

    # ---- lifecycle ----

    def close(self) -> None:
        self.conn.close()
