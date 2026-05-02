"""Decision lifecycle: turn a recorded `propose_fix` into an applied patch.

Plan §3.3 / §8 Day 12:
    - On `propose_fix` (step 2.2): a row lands in the `decisions` table
      with template + params (in `diff_text`).
    - On user approve: read current source → run libcst transformer →
      write new source → git commit on the lensml session branch →
      mark decision approved.
    - On user reject: mark decision rejected, do nothing else.
    - On user undo (after approve): git reset --hard HEAD~1 → mark
      decision undone.

The `diff_text` column overloads — at proposal time it stores a JSON
blob `{"template", "params", "rationale"}`. The actual unified-diff text
is computed on demand (this module) since it depends on the current
state of the script file.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lensml.fixes import build_fix

if TYPE_CHECKING:
    from lensml.server.git import GitRepo

log = logging.getLogger(__name__)


@dataclass
class DiffPreview:
    """Computed view of a pending decision against the current script source."""

    decision_id: int
    template: str
    rationale: str
    params: dict[str, Any]
    old_source: str
    new_source: str | None  # None if the fix didn't apply
    diff: str | None
    applicable: bool
    reason: str | None  # if not applicable, why


@dataclass
class ApplyOutcome:
    applied: bool
    commit: str | None = None
    diff: str | None = None
    reason: str | None = None


def parse_decision_payload(diff_text_field: str | None) -> dict[str, Any]:
    """Decode the JSON blob stored by `propose_fix` in the decision row.

    Tolerant: returns an empty dict on any parse failure (the agent might
    have stored something we don't understand yet).
    """
    if not diff_text_field:
        return {}
    try:
        parsed = json.loads(diff_text_field)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def compute_diff_preview(
    decision: dict[str, Any],
    script_path: Path,
) -> DiffPreview:
    """Read the script, run the fix template, return a DiffPreview."""
    payload = parse_decision_payload(decision.get("diff_text"))
    template = decision.get("fix_template") or payload.get("template", "")
    params = payload.get("params", {}) if isinstance(payload.get("params"), dict) else {}
    rationale = payload.get("rationale", "") if isinstance(payload.get("rationale"), str) else ""

    if not script_path.exists():
        return DiffPreview(
            decision_id=int(decision.get("id", 0)),
            template=template,
            rationale=rationale,
            params=params,
            old_source="",
            new_source=None,
            diff=None,
            applicable=False,
            reason=f"script not found: {script_path}",
        )

    old_source = script_path.read_text(encoding="utf-8")
    try:
        fix = build_fix(template, params)
    except ValueError as e:
        return DiffPreview(
            decision_id=int(decision.get("id", 0)),
            template=template,
            rationale=rationale,
            params=params,
            old_source=old_source,
            new_source=None,
            diff=None,
            applicable=False,
            reason=str(e),
        )

    result = fix.apply(old_source)
    return DiffPreview(
        decision_id=int(decision.get("id", 0)),
        template=template,
        rationale=rationale,
        params=params,
        old_source=old_source,
        new_source=result.new_source if result.applied else None,
        diff=result.diff if result.applied else None,
        applicable=result.applied,
        reason=None if result.applied else result.reason,
    )


def apply_decision(
    decision: dict[str, Any],
    *,
    script_path: Path,
    repo: GitRepo,
    run_id: str,
) -> ApplyOutcome:
    """Apply the decision's fix template to ``script_path`` and commit.

    Caller is expected to have arranged for ``repo`` to be on the run's
    session branch (created by `create_session_branch` at run start).
    """
    preview = compute_diff_preview(decision, script_path)
    if not preview.applicable or preview.new_source is None:
        return ApplyOutcome(applied=False, reason=preview.reason)

    script_path.write_text(preview.new_source, encoding="utf-8")
    template = preview.template or "?"
    msg = f"lensml: {template} (run {run_id} decision {preview.decision_id})"
    try:
        commit = repo.commit_file(script_path, msg)
    except Exception as e:
        log.exception("git commit failed")
        return ApplyOutcome(applied=False, reason=f"git commit failed: {e}")
    return ApplyOutcome(
        applied=True,
        commit=commit,
        diff=preview.diff,
    )
