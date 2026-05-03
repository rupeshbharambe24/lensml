"""Unit tests for lensml.bench.runner."""

from __future__ import annotations

import difflib
import json
import os
from pathlib import Path

import pytest

from lensml.bench.fixtures import discover_fixtures
from lensml.bench.runner import BenchRunner, BenchSummary, FixtureRun

# ---- in-tmp-dir scratch fixture (fast, deterministic) -------------------


def _bug_source(final_loss: float) -> str:
    """A 1-line "training" script that just prints a final-loss JSON line."""
    return f'import json\nprint(json.dumps({{"final_loss": {final_loss}}}))\n'


def _make_scratch_fixture(
    base: Path,
    *,
    pathology: str = "exploding_grad",
    bug_loss: float = 1.5,
    fixed_loss: float = 0.1,
    loss_threshold: float = 0.5,
) -> Path:
    """Create a runnable fixture under base/tiny/<pathology>/0/.

    bug.py prints ``bug_loss``; the canonical fix.diff turns it into a
    script that prints ``fixed_loss``. Loss threshold (in meta.json)
    sits between them so the bug fails and the fix passes.
    """
    p = base / "tiny" / pathology / "0"
    p.mkdir(parents=True, exist_ok=True)
    bug = _bug_source(bug_loss)
    patched = _bug_source(fixed_loss)
    diff = "".join(
        difflib.unified_diff(
            bug.splitlines(keepends=True),
            patched.splitlines(keepends=True),
            fromfile="a/bug.py",
            tofile="b/bug.py",
        )
    )
    (p / "bug.py").write_text(bug, encoding="utf-8")
    (p / "fix.diff").write_text(diff, encoding="utf-8")
    (p / "grade.py").write_text("# stub\n", encoding="utf-8")
    (p / "meta.json").write_text(
        json.dumps(
            {
                "pathology_class": pathology,
                "difficulty": "easy",
                "host_env": "tiny",
                "max_steps": 10,
                "loss_threshold": loss_threshold,
            }
        ),
        encoding="utf-8",
    )
    return p


def _runner_for(tmp_path: Path) -> BenchRunner:
    fixtures = discover_fixtures(tmp_path)
    return BenchRunner(fixtures=fixtures)


# ---- run_one -------------------------------------------------------------


def test_run_one_bug_fails_fix_passes(tmp_path: Path) -> None:
    _make_scratch_fixture(tmp_path)
    runner = _runner_for(tmp_path)
    run = runner.run_one(runner.fixtures[0])
    assert isinstance(run, FixtureRun)
    assert run.bug_correctly_fails is True
    assert run.fix_passes is True
    assert run.overall_pass is True


def test_run_one_records_pathology_and_env(tmp_path: Path) -> None:
    _make_scratch_fixture(tmp_path, pathology="dead_relu")
    runner = _runner_for(tmp_path)
    run = runner.run_one(runner.fixtures[0])
    assert run.pathology_class == "dead_relu"
    assert run.host_env == "tiny"
    assert run.fixture_slug == "tiny/dead_relu/0"


def test_run_one_with_unfixed_bug_records_failure(tmp_path: Path) -> None:
    """If bug.py and patched.py have the same loss, the bug "doesn't fail" —
    the fixture is degenerate and overall_pass should be False."""
    _make_scratch_fixture(tmp_path, bug_loss=0.1, fixed_loss=0.1)
    runner = _runner_for(tmp_path)
    run = runner.run_one(runner.fixtures[0])
    assert run.fix_passes is True
    assert run.bug_correctly_fails is False
    assert run.overall_pass is False


def test_run_one_handles_broken_diff(tmp_path: Path) -> None:
    p = _make_scratch_fixture(tmp_path)
    # Corrupt the diff so apply_diff raises.
    (p / "fix.diff").write_text(
        "--- a/bug.py\n+++ b/bug.py\n@@ -1,1 +1,1 @@\n-totally-different\n+other\n",
        encoding="utf-8",
    )
    runner = _runner_for(tmp_path)
    run = runner.run_one(runner.fixtures[0])
    assert run.fix_passes is False
    assert "diff did not apply" in (run.patched.reason or "")


# ---- run_all -------------------------------------------------------------


def test_run_all_summary_aggregates(tmp_path: Path) -> None:
    _make_scratch_fixture(tmp_path, pathology="exploding_grad")
    _make_scratch_fixture(tmp_path, pathology="dead_relu")
    summary = _runner_for(tmp_path).run_all()
    assert isinstance(summary, BenchSummary)
    assert summary.total == 2
    assert summary.overall_pass == 2
    assert summary.fix_pass == 2
    assert summary.bug_correctly_fails == 2
    assert summary.fix_pass_rate == 1.0
    assert summary.overall_pass_rate == 1.0


def test_run_all_filter_by_env(tmp_path: Path) -> None:
    _make_scratch_fixture(tmp_path, pathology="exploding_grad")
    summary = _runner_for(tmp_path).run_all(env="tiny")
    assert summary.total == 1
    summary2 = _runner_for(tmp_path).run_all(env="small")
    assert summary2.total == 0


def test_run_all_filter_by_pathology(tmp_path: Path) -> None:
    _make_scratch_fixture(tmp_path, pathology="exploding_grad")
    _make_scratch_fixture(tmp_path, pathology="dead_relu")
    summary = _runner_for(tmp_path).run_all(pathology="dead_relu")
    assert summary.total == 1
    assert summary.runs[0].pathology_class == "dead_relu"


def test_run_all_zero_total_safe(tmp_path: Path) -> None:
    summary = _runner_for(tmp_path).run_all()
    assert summary.total == 0
    assert summary.fix_pass_rate == 0.0


# ---- caching -------------------------------------------------------------


def test_cache_round_trip(tmp_path: Path) -> None:
    _make_scratch_fixture(tmp_path)
    cache_path = tmp_path / "cache.json"
    runner = BenchRunner(fixtures=discover_fixtures(tmp_path), cache_path=cache_path)
    s1 = runner.run_all()
    assert s1.overall_pass == 1
    assert cache_path.exists()
    # Tweak bug.py to print stale info — cached entry should still be reused
    # (we trust the previous overall_pass).
    bug = next(p for p in tmp_path.rglob("bug.py"))
    bug.write_text("raise SystemExit(1)\n", encoding="utf-8")
    runner2 = BenchRunner(fixtures=discover_fixtures(tmp_path), cache_path=cache_path)
    s2 = runner2.run_all()
    assert s2.overall_pass == 1


def test_force_bypasses_cache(tmp_path: Path) -> None:
    _make_scratch_fixture(tmp_path)
    cache_path = tmp_path / "cache.json"
    runner = BenchRunner(fixtures=discover_fixtures(tmp_path), cache_path=cache_path)
    runner.run_all()
    bug = next(p for p in tmp_path.rglob("bug.py"))
    bug.write_text("raise SystemExit(1)\n", encoding="utf-8")
    runner2 = BenchRunner(fixtures=discover_fixtures(tmp_path), cache_path=cache_path)
    s = runner2.run_all(force=True)
    assert s.overall_pass == 0


def test_cache_unreadable_starts_fresh(tmp_path: Path) -> None:
    _make_scratch_fixture(tmp_path)
    cache_path = tmp_path / "cache.json"
    cache_path.write_text("not-json", encoding="utf-8")
    runner = BenchRunner(fixtures=discover_fixtures(tmp_path), cache_path=cache_path)
    s = runner.run_all()  # must not raise
    assert s.total == 1


# ---- end-to-end against a real packaged fixture --------------------------


@pytest.mark.skipif(
    os.environ.get("LENSML_BENCH_E2E") != "1",
    reason="Heavy: runs real PyTorch fixtures end-to-end. Opt in with LENSML_BENCH_E2E=1.",
)
def test_packaged_exploding_grad_runs_end_to_end() -> None:
    """End-to-end smoke test against the exploding_grad tiny fixture.

    Skipped by default (it spawns 2 PyTorch subprocesses ≈ 5-10 s of CPU)
    but kept here so step 4.3 has the integration check the plan asks
    for. Run with ``LENSML_BENCH_E2E=1 pytest -k packaged_exploding_grad``.

    Why this fixture: lr=0.5 + no clip + amplified inputs reliably drive
    loss to NaN, so ``bug_correctly_fails`` is robust. Other fixtures
    (lr_too_high, dead_relu) sometimes converge anyway on tiny synthetic
    data — they're still useful as agent-bench inputs but flakier as
    runner integration checks.
    """
    from lensml.bench.runner import BenchRunner

    runner = BenchRunner(timeout=120.0)
    summary = runner.run_all(pathology="exploding_grad", env="tiny")
    assert summary.total == 1
    run = summary.runs[0]
    assert run.fix_passes, run.patched.reason
    assert run.bug_correctly_fails, "bug.py should not converge"
