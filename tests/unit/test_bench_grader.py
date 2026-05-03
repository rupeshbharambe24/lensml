"""Unit tests for lensml.bench.grader."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from lensml.bench.fixtures import Fixture, FixtureMeta, load_fixture
from lensml.bench.grader import (
    GradeChecks,
    GradeResult,
    apply_diff,
    grade_script,
    run_script,
)


def _make_fixture(tmp_path: Path, **meta_overrides) -> Fixture:
    meta = FixtureMeta(
        pathology_class="exploding_grad",
        difficulty="easy",
        host_env="tiny",
        max_steps=100,
        loss_threshold=0.5,
        metric_threshold=None,
    )
    for k, v in meta_overrides.items():
        setattr(meta, k, v)
    return Fixture(path=tmp_path, meta=meta, bug_source="", fix_diff="", raw_meta={})


# ---- run_script ----


def test_run_script_returns_exit_code_and_streams(tmp_path: Path) -> None:
    p = tmp_path / "ok.py"
    p.write_text("print('hi'); import sys; sys.stderr.write('err')", encoding="utf-8")
    code, stdout, stderr, elapsed = run_script(p, timeout=10)
    assert code == 0
    assert "hi" in stdout
    assert "err" in stderr
    assert elapsed >= 0


def test_run_script_nonzero_exit(tmp_path: Path) -> None:
    p = tmp_path / "fail.py"
    p.write_text("import sys; sys.exit(2)", encoding="utf-8")
    code, _, _, _ = run_script(p, timeout=10)
    assert code == 2


def test_run_script_timeout(tmp_path: Path) -> None:
    p = tmp_path / "slow.py"
    p.write_text("import time; time.sleep(5)", encoding="utf-8")
    code, _, stderr, _ = run_script(p, timeout=0.5)
    assert code == -1
    assert "timed out" in stderr


# ---- grade_script ----


def test_grade_passes_when_loss_below_threshold(tmp_path: Path) -> None:
    p = tmp_path / "good.py"
    p.write_text("import json; print(json.dumps({'final_loss': 0.1}))", encoding="utf-8")
    fix = _make_fixture(tmp_path)
    result = grade_script(p, fix, variant="patched")
    assert result.checks.runs is True
    assert result.checks.loss_converged is True
    assert result.passed is True
    assert result.final_loss == 0.1


def test_grade_fails_when_loss_above_threshold(tmp_path: Path) -> None:
    p = tmp_path / "bad.py"
    p.write_text("import json; print(json.dumps({'final_loss': 1.5}))", encoding="utf-8")
    fix = _make_fixture(tmp_path)
    result = grade_script(p, fix, variant="bug")
    assert result.checks.runs is True
    assert result.checks.loss_converged is False
    assert result.passed is False


def test_grade_fails_when_script_errors(tmp_path: Path) -> None:
    p = tmp_path / "crash.py"
    p.write_text("raise RuntimeError('boom')", encoding="utf-8")
    fix = _make_fixture(tmp_path)
    result = grade_script(p, fix, variant="bug")
    assert result.checks.runs is False
    assert result.passed is False
    assert "exit_code" in (result.reason or "")


def test_grade_metric_check(tmp_path: Path) -> None:
    p = tmp_path / "with_metric.py"
    p.write_text(
        "import json; print(json.dumps({'final_loss': 0.1, 'final_metric': 0.92}))",
        encoding="utf-8",
    )
    fix = _make_fixture(tmp_path, metric_threshold=0.9)
    result = grade_script(p, fix, variant="patched")
    assert result.checks.metric_passed is True
    assert result.passed is True


def test_grade_metric_below_threshold(tmp_path: Path) -> None:
    p = tmp_path / "low_metric.py"
    p.write_text(
        "import json; print(json.dumps({'final_loss': 0.1, 'final_metric': 0.5}))",
        encoding="utf-8",
    )
    fix = _make_fixture(tmp_path, metric_threshold=0.9)
    result = grade_script(p, fix, variant="patched")
    assert result.checks.metric_passed is False
    assert result.passed is False


def test_grade_no_status_line(tmp_path: Path) -> None:
    p = tmp_path / "silent.py"
    p.write_text("print('nothing useful')", encoding="utf-8")
    fix = _make_fixture(tmp_path)
    result = grade_script(p, fix, variant="bug")
    assert result.checks.runs is True
    assert result.checks.loss_converged is False
    assert "no JSON status line" in (result.reason or "")


def test_grade_result_to_dict_includes_passed() -> None:
    r = GradeResult(
        fixture_slug="x",
        variant="bug",
        checks=GradeChecks(runs=True, loss_converged=True),
    )
    d = r.to_dict()
    assert d["passed"] is True
    assert d["fixture_slug"] == "x"


# ---- apply_diff ----


def test_apply_diff_simple_replace() -> None:
    src = "a = 1\nb = 2\nc = 3\n"
    diff = """--- a/file.py
+++ b/file.py
@@ -1,3 +1,3 @@
 a = 1
-b = 2
+b = 20
 c = 3
"""
    out = apply_diff(src, diff)
    assert out == "a = 1\nb = 20\nc = 3\n"


def test_apply_diff_pure_insertion() -> None:
    src = "a = 1\nb = 2\n"
    diff = """--- a/file.py
+++ b/file.py
@@ -1,2 +1,3 @@
 a = 1
+x = 99
 b = 2
"""
    out = apply_diff(src, diff)
    assert out == "a = 1\nx = 99\nb = 2\n"


def test_apply_diff_no_hunks_is_identity() -> None:
    src = "a = 1\n"
    out = apply_diff(src, "")
    assert out == src


def test_apply_diff_context_mismatch_raises() -> None:
    src = "totally\nunrelated\nfile\n"
    diff = """--- a/file.py
+++ b/file.py
@@ -1,3 +1,3 @@
 a = 1
-b = 2
+b = 20
 c = 3
"""
    with pytest.raises(ValueError):
        apply_diff(src, diff)


# ---- end-to-end: packaged fixtures patch cleanly ----


@pytest.mark.parametrize(
    "slug",
    [
        "tiny/exploding_grad/0",
        "tiny/dead_relu/0",
        "tiny/missing_normalize/0",
        "tiny/lr_too_high/0",
        "tiny/bad_init/0",
        "tiny/vanishing_grad/0",
        "tiny/model_eval_not_called/0",
        "tiny/lr_schedule_mismatch/0",
    ],
)
def test_packaged_fixture_diff_applies_cleanly(slug: str) -> None:
    """Every shipped fixture's fix.diff must apply cleanly to its bug.py
    and the result must be parseable Python."""
    import ast

    parts = slug.split("/")
    root = Path(__file__).resolve().parents[2] / "lensml" / "bench" / "fixtures"
    fix = load_fixture(root / parts[0] / parts[1] / parts[2])
    patched = apply_diff(fix.bug_source, fix.fix_diff)
    assert patched != fix.bug_source, f"{slug}: fix.diff produced no change"
    ast.parse(patched)  # parses cleanly


def _ensure_repo_on_path() -> None:
    """The fixture grade.py scripts ``import lensml.bench.*`` — make sure
    they can find the package when invoked as a subprocess."""
    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))


_ensure_repo_on_path()
