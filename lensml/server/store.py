"""SQLite event log.

Plan §3.3 — schema (raw SQL, no SQLAlchemy):
    runs(id, started_at, ended_at, script_path, git_branch, status)
    events(id, ts, run_id, kind, payload_json)              -- append-only
    decisions(id, run_id, ts, fix_template, diff_text, approved, applied_at)

Path: `~/.lensml/runs/{run_id}/events.db`.
"""

from __future__ import annotations

# Implemented Day 3.
