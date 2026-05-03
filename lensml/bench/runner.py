"""Benchmark runner — orchestrates fixture execution (plan §6, §8 D24).

For each fixture:
    1. Run ``bug.py`` in a subprocess. Grade. Expected: fails ≥ one check.
    2. Apply ``fix.diff`` to ``bug.py`` to produce ``patched.py`` (in a
       per-fixture temp dir). Run. Grade. Expected: passes all checks.
    3. Record both ``GradeResult`` objects into a paired ``FixtureRun``.

Idempotent: ``run_all(..., force=False)`` skips fixtures whose previous
``patched`` run passed (loaded from a cache file). Pass ``force=True``
to re-run everything.

Plan §11 risk #6 (Colab/Kaggle disconnects): the cache lives on disk
and the runner is happy to resume after a crash.

The runner is sync today. Parallelism within a single run can land
later — bench fixtures are network-isolated subprocesses so it's
embarrassingly parallel.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from lensml.bench.fixtures import discover_fixtures
from lensml.bench.grader import (
    DEFAULT_TIMEOUT,
    GradeChecks,
    GradeResult,
    apply_diff,
    grade_script,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from lensml.bench.fixtures import Fixture

log = logging.getLogger(__name__)


@dataclass
class FixtureRun:
    """Result of running both variants (bug + patched) of one fixture."""

    fixture_slug: str
    pathology_class: str
    host_env: str
    bug: GradeResult
    patched: GradeResult
    elapsed: float = 0.0

    @property
    def fix_passes(self) -> bool:
        """The patched script passes the grader."""
        return self.patched.passed

    @property
    def bug_correctly_fails(self) -> bool:
        """The bug script fails (otherwise the fixture isn't really a bug)."""
        return not self.bug.passed

    @property
    def overall_pass(self) -> bool:
        """A fixture is a clean signal only if the bug fails AND the fix passes."""
        return self.fix_passes and self.bug_correctly_fails

    def to_dict(self) -> dict:
        return {
            "fixture_slug": self.fixture_slug,
            "pathology_class": self.pathology_class,
            "host_env": self.host_env,
            "elapsed": self.elapsed,
            "fix_passes": self.fix_passes,
            "bug_correctly_fails": self.bug_correctly_fails,
            "overall_pass": self.overall_pass,
            "bug": self.bug.to_dict(),
            "patched": self.patched.to_dict(),
        }


@dataclass
class BenchSummary:
    """Aggregate stats from a ``BenchRunner.run_all`` call."""

    total: int = 0
    fix_pass: int = 0
    bug_correctly_fails: int = 0
    overall_pass: int = 0
    elapsed: float = 0.0
    runs: list[FixtureRun] = field(default_factory=list)

    @property
    def fix_pass_rate(self) -> float:
        return self.fix_pass / self.total if self.total else 0.0

    @property
    def overall_pass_rate(self) -> float:
        return self.overall_pass / self.total if self.total else 0.0

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "fix_pass": self.fix_pass,
            "bug_correctly_fails": self.bug_correctly_fails,
            "overall_pass": self.overall_pass,
            "fix_pass_rate": self.fix_pass_rate,
            "overall_pass_rate": self.overall_pass_rate,
            "elapsed": self.elapsed,
            "runs": [r.to_dict() for r in self.runs],
        }


class BenchRunner:
    """Subprocess-based fixture orchestrator (plan §6 / §8 D24)."""

    def __init__(
        self,
        *,
        fixtures: list[Fixture] | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        cache_path: Path | None = None,
    ) -> None:
        """If ``fixtures`` is None we ``discover_fixtures()`` from the
        in-package directory."""
        self.fixtures = fixtures if fixtures is not None else discover_fixtures()
        self.timeout = timeout
        self.cache_path = cache_path

    # ---- caching ----

    def _load_cache(self) -> dict[str, dict]:
        if self.cache_path is None or not self.cache_path.exists():
            return {}
        try:
            return json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            log.warning("bench: cache file %s unreadable; starting fresh", self.cache_path)
            return {}

    def _save_cache(self, cache: dict[str, dict]) -> None:
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(cache, default=str, indent=2), encoding="utf-8")

    # ---- single-fixture run ----

    def run_one(self, fixture: Fixture) -> FixtureRun:
        """Run bug + patched variants of a single fixture. Never raises —
        a crash inside the subprocess shows up in the GradeResult."""
        t0 = time.time()
        with tempfile.TemporaryDirectory(prefix="lensml-bench-") as tmp:
            tmp_path = Path(tmp)
            # Copy bug.py + supporting files (grade.py, meta.json) into tmp so
            # the patched script can reference them if needed.
            for fname in ("bug.py", "fix.diff", "grade.py", "meta.json"):
                shutil.copy2(fixture.path / fname, tmp_path / fname)

            # bug variant runs the original script in place — we use the
            # in-package path so any relative imports / assets keep working.
            bug_result = grade_script(
                fixture.path / "bug.py", fixture, variant="bug", timeout=self.timeout
            )

            # patched variant: apply the diff and run it in tmp.
            try:
                patched_src = apply_diff(fixture.bug_source, fixture.fix_diff)
            except ValueError as e:
                # Diff didn't apply cleanly — fail patched grade but record.
                patched_result = GradeResult(
                    fixture_slug=fixture.slug,
                    variant="patched",
                    checks=GradeChecks(),
                    reason=f"diff did not apply: {e}",
                )
                return FixtureRun(
                    fixture_slug=fixture.slug,
                    pathology_class=fixture.meta.pathology_class,
                    host_env=fixture.meta.host_env,
                    bug=bug_result,
                    patched=patched_result,
                    elapsed=time.time() - t0,
                )

            patched_path = tmp_path / "patched.py"
            patched_path.write_text(patched_src, encoding="utf-8")
            patched_result = grade_script(
                patched_path, fixture, variant="patched", timeout=self.timeout
            )

        return FixtureRun(
            fixture_slug=fixture.slug,
            pathology_class=fixture.meta.pathology_class,
            host_env=fixture.meta.host_env,
            bug=bug_result,
            patched=patched_result,
            elapsed=time.time() - t0,
        )

    # ---- top level ----

    def run_all(
        self,
        *,
        env: str | None = None,
        pathology: str | None = None,
        force: bool = False,
    ) -> BenchSummary:
        """Run every fixture matching the filters. Skips fixtures whose
        cached ``overall_pass`` is True unless ``force=True``.

        Filters:
            env: only run fixtures whose ``host_env == env``.
            pathology: only run fixtures whose ``pathology_class`` matches.
        """
        selected = list(self._filter(env=env, pathology=pathology))
        cache = self._load_cache() if not force else {}
        summary = BenchSummary()
        t0 = time.time()
        for fix in selected:
            cached = cache.get(fix.slug)
            if not force and cached and cached.get("overall_pass") is True:
                # Replay the cached result so we report consistent numbers.
                run = _from_cached(cached)
            else:
                log.info("bench: running %s", fix.slug)
                run = self.run_one(fix)
                cache[fix.slug] = run.to_dict()
            summary.runs.append(run)
            summary.total += 1
            if run.fix_passes:
                summary.fix_pass += 1
            if run.bug_correctly_fails:
                summary.bug_correctly_fails += 1
            if run.overall_pass:
                summary.overall_pass += 1

        summary.elapsed = time.time() - t0
        self._save_cache(cache)
        return summary

    def _filter(self, *, env: str | None, pathology: str | None) -> Iterable[Fixture]:
        for fix in self.fixtures:
            if env is not None and fix.meta.host_env != env:
                continue
            if pathology is not None and fix.meta.pathology_class != pathology:
                continue
            yield fix


def _from_cached(d: dict) -> FixtureRun:
    """Reconstruct a FixtureRun from a previously-serialized cache entry.

    Best-effort — if the schema drifted, individual fields fall back to
    sensible defaults rather than raising.
    """
    bug_d = d.get("bug", {})
    patched_d = d.get("patched", {})
    bug_checks = bug_d.get("checks") or {}
    patched_checks = patched_d.get("checks") or {}

    bug = GradeResult(
        fixture_slug=bug_d.get("fixture_slug", d.get("fixture_slug", "")),
        variant=bug_d.get("variant", "bug"),
        checks=GradeChecks(
            runs=bool(bug_checks.get("runs", False)),
            loss_converged=bool(bug_checks.get("loss_converged", False)),
            metric_passed=bug_checks.get("metric_passed"),
        ),
        final_loss=bug_d.get("final_loss"),
        final_metric=bug_d.get("final_metric"),
        steps_used=bug_d.get("steps_used"),
        elapsed=float(bug_d.get("elapsed", 0.0) or 0.0),
        exit_code=bug_d.get("exit_code"),
        reason=bug_d.get("reason"),
    )
    patched = GradeResult(
        fixture_slug=patched_d.get("fixture_slug", d.get("fixture_slug", "")),
        variant=patched_d.get("variant", "patched"),
        checks=GradeChecks(
            runs=bool(patched_checks.get("runs", False)),
            loss_converged=bool(patched_checks.get("loss_converged", False)),
            metric_passed=patched_checks.get("metric_passed"),
        ),
        final_loss=patched_d.get("final_loss"),
        final_metric=patched_d.get("final_metric"),
        steps_used=patched_d.get("steps_used"),
        elapsed=float(patched_d.get("elapsed", 0.0) or 0.0),
        exit_code=patched_d.get("exit_code"),
        reason=patched_d.get("reason"),
    )
    return FixtureRun(
        fixture_slug=d.get("fixture_slug", ""),
        pathology_class=d.get("pathology_class", ""),
        host_env=d.get("host_env", ""),
        bug=bug,
        patched=patched,
        elapsed=float(d.get("elapsed", 0.0) or 0.0),
    )


__all__ = [
    "BenchRunner",
    "BenchSummary",
    "FixtureRun",
]
