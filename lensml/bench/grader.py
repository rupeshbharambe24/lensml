"""Deterministic per-fixture grader (plan §6.1, §6.4).

Three checks per fixture (plan §6.1):
    (a) does the (patched) script run without error?
    (b) does loss converge below ``meta.loss_threshold`` within ``max_steps``?
    (c) does the test metric exceed ``meta.metric_threshold`` (if set)?

The grader runs the candidate script in a subprocess so a crash or
``sys.exit`` can't take down the runner. The script emits a final-line
status JSON like::

    {"final_loss": 0.143, "final_metric": 0.92, "steps_used": 200}

…which we parse from the last line of stdout. Any earlier non-JSON
stdout is preserved in the result for debugging.

The patcher applies a ``fix.diff`` to ``bug.py`` using a minimal hunk
walker that understands the output of ``difflib.unified_diff`` — no
external ``patch`` binary needed.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from lensml.bench.fixtures import Fixture

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 120.0  # tiny fixtures must finish in 30-90 s; cushion to 120 s


@dataclass
class GradeChecks:
    runs: bool = False
    loss_converged: bool = False
    metric_passed: bool | None = None  # None when no metric threshold defined


@dataclass
class GradeResult:
    """Outcome of grading a single fixture run."""

    fixture_slug: str
    variant: str  # "bug" | "patched"
    checks: GradeChecks
    final_loss: float | None = None
    final_metric: float | None = None
    steps_used: int | None = None
    elapsed: float = 0.0
    exit_code: int | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        c = self.checks
        if not c.runs or not c.loss_converged:
            return False
        return c.metric_passed is not False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["passed"] = self.passed
        return d


def _parse_final_status(stdout: str) -> dict[str, Any] | None:
    """Pull the LAST JSON object printed on stdout (the grader contract)."""
    for raw in reversed(stdout.splitlines()):
        line = raw.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _tail(s: str, max_chars: int = 1500) -> str:
    if len(s) <= max_chars:
        return s
    return s[-max_chars:]


def run_script(
    script: Path,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    env: dict[str, str] | None = None,
    python: str | None = None,
) -> tuple[int, str, str, float]:
    """Subprocess a Python script. Returns (exit_code, stdout, stderr, elapsed)."""
    interp = python or sys.executable
    cmd = [interp, str(script)]
    proc_env = os.environ.copy()
    if env:
        proc_env.update(env)
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=proc_env,
            check=False,
        )
        return proc.returncode, proc.stdout, proc.stderr, time.time() - t0
    except subprocess.TimeoutExpired as e:
        return (
            -1,
            (e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")),
            (e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or ""))
            + f"\n[grader] timed out after {timeout}s",
            time.time() - t0,
        )


def grade_script(
    script: Path,
    fixture: Fixture,
    *,
    variant: str,
    timeout: float = DEFAULT_TIMEOUT,
    python: str | None = None,
) -> GradeResult:
    """Run ``script`` and apply the three plan §6.1 checks."""
    code, stdout, stderr, elapsed = run_script(script, timeout=timeout, python=python)
    checks = GradeChecks()

    runs = code == 0
    checks.runs = runs

    final_status = _parse_final_status(stdout) if runs else None
    final_loss: float | None = None
    final_metric: float | None = None
    steps_used: int | None = None
    if isinstance(final_status, dict):
        v = final_status.get("final_loss")
        if isinstance(v, (int, float)):
            final_loss = float(v)
        v = final_status.get("final_metric")
        if isinstance(v, (int, float)):
            final_metric = float(v)
        v = final_status.get("steps_used")
        if isinstance(v, int):
            steps_used = v

    if final_loss is not None and final_loss <= fixture.meta.loss_threshold:
        checks.loss_converged = True
    else:
        checks.loss_converged = False

    if fixture.meta.metric_threshold is not None:
        if final_metric is None:
            checks.metric_passed = False
        else:
            checks.metric_passed = final_metric >= fixture.meta.metric_threshold

    reason: str | None = None
    if not runs:
        reason = f"exit_code={code}"
    elif final_status is None:
        reason = "no JSON status line on stdout"

    return GradeResult(
        fixture_slug=fixture.slug,
        variant=variant,
        checks=checks,
        final_loss=final_loss,
        final_metric=final_metric,
        steps_used=steps_used,
        elapsed=elapsed,
        exit_code=code,
        stdout_tail=_tail(stdout),
        stderr_tail=_tail(stderr),
        reason=reason,
    )


def apply_diff(source: str, diff: str) -> str:
    """Apply a unified diff to ``source``. Returns the patched text.

    Minimal hunk-based applier: walks the diff, finds each hunk by its
    ``@@ -a,b +c,d @@`` header, and substitutes the new lines. Does NOT
    support multi-file diffs or fuzzy matching. Raises ``ValueError`` on
    a hunk that doesn't apply cleanly.
    """
    src_lines = source.splitlines(keepends=True)
    out_lines: list[str] = []
    src_idx = 0

    diff_lines = diff.splitlines(keepends=True)
    i = 0
    while i < len(diff_lines) and not diff_lines[i].startswith("@@"):
        i += 1
    if i >= len(diff_lines):
        return source  # no hunks → identity

    while i < len(diff_lines):
        line = diff_lines[i]
        if not line.startswith("@@"):
            i += 1
            continue
        try:
            old_part = line.split(" ")[1]
            old_start = int(old_part.split(",")[0].lstrip("-"))
        except (IndexError, ValueError) as e:
            raise ValueError(f"malformed hunk header: {line!r}") from e
        hunk_src_idx = old_start - 1

        while src_idx < hunk_src_idx and src_idx < len(src_lines):
            out_lines.append(src_lines[src_idx])
            src_idx += 1

        i += 1
        while i < len(diff_lines) and not diff_lines[i].startswith("@@"):
            hl = diff_lines[i]
            if hl.startswith(" "):
                if src_idx >= len(src_lines) or src_lines[src_idx] != hl[1:]:
                    raise ValueError(f"hunk context mismatch at source line {src_idx + 1}")
                out_lines.append(src_lines[src_idx])
                src_idx += 1
            elif hl.startswith("-"):
                if src_idx >= len(src_lines) or src_lines[src_idx] != hl[1:]:
                    raise ValueError(f"hunk delete mismatch at source line {src_idx + 1}")
                src_idx += 1
            elif hl.startswith("+"):
                out_lines.append(hl[1:])
            i += 1

    while src_idx < len(src_lines):
        out_lines.append(src_lines[src_idx])
        src_idx += 1

    return "".join(out_lines)


__all__ = [
    "DEFAULT_TIMEOUT",
    "GradeChecks",
    "GradeResult",
    "apply_diff",
    "grade_script",
    "run_script",
]
