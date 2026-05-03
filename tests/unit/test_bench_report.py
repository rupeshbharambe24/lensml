"""Unit tests for lensml.bench.report."""

from __future__ import annotations

from pathlib import Path

from lensml.bench.grader import GradeChecks, GradeResult
from lensml.bench.report import build_report, render_markdown
from lensml.bench.runner import BenchSummary, FixtureRun


def _run(slug: str, *, fix_passes: bool, bug_fails: bool) -> FixtureRun:
    bug_checks = GradeChecks(runs=True, loss_converged=not bug_fails)
    bug = GradeResult(fixture_slug=slug, variant="bug", checks=bug_checks)
    patched = GradeResult(
        fixture_slug=slug,
        variant="patched",
        checks=GradeChecks(runs=True, loss_converged=fix_passes),
    )
    pathology = slug.split("/")[1]
    env = slug.split("/")[0]
    return FixtureRun(
        fixture_slug=slug,
        pathology_class=pathology,
        host_env=env,
        bug=bug,
        patched=patched,
    )


def _summary(*runs: FixtureRun) -> BenchSummary:
    s = BenchSummary()
    for r in runs:
        s.runs.append(r)
        s.total += 1
        if r.fix_passes:
            s.fix_pass += 1
        if r.bug_correctly_fails:
            s.bug_correctly_fails += 1
        if r.overall_pass:
            s.overall_pass += 1
    s.elapsed = 1.5
    return s


def test_render_markdown_includes_headline() -> None:
    s = _summary(
        _run("tiny/exploding_grad/0", fix_passes=True, bug_fails=True),
        _run("tiny/dead_relu/0", fix_passes=True, bug_fails=True),
    )
    md = render_markdown(s)
    assert "lensml-bench report" in md
    assert "Overall pass:** 2/2" in md
    assert "100.0%" in md


def test_render_markdown_lists_failures() -> None:
    s = _summary(
        _run("tiny/exploding_grad/0", fix_passes=True, bug_fails=True),
        _run("tiny/dead_relu/0", fix_passes=False, bug_fails=True),
    )
    md = render_markdown(s)
    assert "What doesn't work yet" in md
    assert "tiny/dead_relu/0" in md
    assert "patched.py failed grader" in md


def test_render_markdown_all_pass_says_so() -> None:
    s = _summary(_run("tiny/exploding_grad/0", fix_passes=True, bug_fails=True))
    md = render_markdown(s)
    assert "All fixtures pass cleanly" in md


def test_per_pathology_table_aggregates() -> None:
    s = _summary(
        _run("tiny/exploding_grad/0", fix_passes=True, bug_fails=True),
        _run("tiny/exploding_grad/1", fix_passes=False, bug_fails=True),
        _run("tiny/dead_relu/0", fix_passes=True, bug_fails=True),
    )
    md = render_markdown(s)
    # exploding_grad has 1/2 overall pass, dead_relu has 1/1.
    assert "exploding_grad" in md
    assert "1/2" in md


def test_build_report_writes_files(tmp_path: Path) -> None:
    s = _summary(_run("tiny/exploding_grad/0", fix_passes=True, bug_fails=True))
    report = build_report(s)
    md_path, json_path = report.write(tmp_path / "out")
    assert md_path.exists()
    assert json_path.exists()
    assert "Overall pass" in md_path.read_text(encoding="utf-8")
    import json as _json

    blob = _json.loads(json_path.read_text(encoding="utf-8"))
    assert blob["total"] == 1
    assert blob["overall_pass"] == 1


def test_render_markdown_empty_summary() -> None:
    s = _summary()  # no runs
    md = render_markdown(s)
    assert "(no fixtures)" in md


def test_render_markdown_with_baselines_includes_comparison_table() -> None:
    """Plan §6.3 — LLM-only baseline comparison must show up in the
    report when baselines are passed."""
    from lensml.bench.baseline import BaselineSummary

    s = _summary(_run("tiny/exploding_grad/0", fix_passes=True, bug_fails=True))
    baseline = BaselineSummary(provider="groq", total=1, diagnosis_correct=1, fix_passes=0)
    md = render_markdown(s, baselines=[baseline])
    assert "lensml-full vs LLM-only baselines" in md
    assert "credibility lynchpin" in md
    assert "lensml-full" in md
    assert "groq" in md


def test_build_report_persists_baselines_in_json(tmp_path: Path) -> None:
    from lensml.bench.baseline import BaselineSummary
    from lensml.bench.report import build_report

    s = _summary(_run("tiny/exploding_grad/0", fix_passes=True, bug_fails=True))
    baseline = BaselineSummary(provider="cerebras", total=1, diagnosis_correct=0, fix_passes=0)
    report = build_report(s, baselines=[baseline])
    md_path, json_path = report.write(tmp_path / "out")
    assert md_path.exists()
    import json as _json

    blob = _json.loads(json_path.read_text(encoding="utf-8"))
    assert "baselines" in blob
    assert blob["baselines"][0]["provider"] == "cerebras"
