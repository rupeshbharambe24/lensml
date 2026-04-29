"""Git branch / commit / reset operations.

Plan §3.3:
    - On `lensml.watch()` start: ensure user is in a git repo (else create
      `.lensml-shadow` repo for the script's directory). Create branch
      `lensml/session-{ts}`.
    - On approved fix apply: commit with message
      `lensml: {fix_template} (run {run_id} step {step})`.
    - On reject: no-op.
    - On undo: `git reset --hard HEAD~1`.
    - On session end: branch persists; user merges or discards manually.
"""

from __future__ import annotations

# Implemented Day 12.
