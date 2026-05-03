"""Unit tests for the new plan §6.5 report sections — confusion
matrix, latency histogram, cost breakdown."""

from __future__ import annotations

import json
from pathlib import Path

from lensml.bench.baseline import BaselineResult, BaselineSummary
from lensml.bench.grader import GradeChecks, GradeResult
from lensml.bench.report import (
    _confusion_matrix,
    _render_confusion_matrix,
    _render_cost_breakdown,
    _render_latency_histogram,
    build_report,
    render_markdown,
)
from lensml.bench.runner import BenchSummary, FixtureRun


def _bench_run(slug: str, *, fix_passes: bool, bug_fails: bool, elapsed: float = 0.5) -> FixtureRun:
    bug = GradeResult(
        fixture_slug=slug,
        variant="bug",
        checks=GradeChecks(runs=True, loss_converged=not bug_fails),
    )
    patched = GradeResult(
        fixture_slug=slug,
        variant="patched",
        checks=GradeChecks(runs=True, loss_converged=fix_passes),
    )
    parts = slug.split("/")
    return FixtureRun(
        fixture_slug=slug,
        pathology_class=parts[1],
        host_env=parts[0],
        bug=bug,
        patched=patched,
        elapsed=elapsed,
    )


def _baseline_result(
    *,
    actual: str,
    predicted: str | None,
    fix_passes: bool,
    elapsed: float = 1.0,
) -> BaselineResult:
    return BaselineResult(
        fixture_slug=f"tiny/{actual}/0",
        pathology_class=actual,
        host_env="tiny",
        diagnosed_class=predicted,
        diagnosis_correct=predicted == actual,
        proposed_diff="",
        diff_applied=fix_passes,
        patched=GradeResult(
            fixture_slug=f"tiny/{actual}/0",
            variant="baseline",
            checks=GradeChecks(runs=True, loss_converged=fix_passes),
        ),
        raw_response="",
        elapsed=elapsed,
    )


def _bench_summary(*runs: FixtureRun) -> BenchSummary:
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
    return s


def _baseline_summary(provider: str, *results: BaselineResult) -> BaselineSummary:
    b = BaselineSummary(provider=provider)
    for r in results:
        b.results.append(r)
        b.total += 1
        if r.diagnosis_correct:
            b.diagnosis_correct += 1
        if r.fix_passes:
            b.fix_passes += 1
        if r.diff_applied:
            b.diff_applied += 1
    return b


# ---- confusion matrix ----


def test_confusion_matrix_counts_correctly() -> None:
    b = _baseline_summary(
        "groq",
        _baseline_result(actual="exploding_grad", predicted="exploding_grad", fix_passes=True),
        _baseline_result(actual="exploding_grad", predicted="dead_relu", fix_passes=False),
        _baseline_result(actual="dead_relu", predicted="dead_relu", fix_passes=True),
        _baseline_result(actual="dead_relu", predicted=None, fix_passes=False),
    )
    matrix = _confusion_matrix(b)
    assert matrix["exploding_grad"]["exploding_grad"] == 1
    assert matrix["exploding_grad"]["dead_relu"] == 1
    assert matrix["dead_relu"]["dead_relu"] == 1
    assert matrix["dead_relu"]["(unparsed)"] == 1


def test_render_confusion_matrix_bolds_diagonal() -> None:
    b = _baseline_summary(
        "groq",
        _baseline_result(actual="exploding_grad", predicted="exploding_grad", fix_passes=True),
    )
    md = _render_confusion_matrix(b)
    assert "exploding_grad" in md
    assert "**1**" in md  # diagonal bolded


def test_render_confusion_matrix_empty() -> None:
    b = _baseline_summary("groq")  # no results
    md = _render_confusion_matrix(b)
    assert "(no baseline runs)" in md


# ---- latency histogram ----


def test_latency_histogram_renders_buckets() -> None:
    bench = _bench_summary(
        _bench_run("tiny/exploding_grad/0", fix_passes=True, bug_fails=True, elapsed=0.5),
        _bench_run("tiny/dead_relu/0", fix_passes=True, bug_fails=True, elapsed=2.0),
    )
    baselines = [
        _baseline_summary(
            "groq",
            _baseline_result(
                actual="exploding_grad", predicted="exploding_grad", fix_passes=True, elapsed=1.0
            ),
        )
    ]
    md = _render_latency_histogram(bench, baselines)
    assert "lensml-full" in md
    assert "baseline·groq" in md
    assert "```" in md  # rendered as a code block


def test_latency_histogram_with_no_runs() -> None:
    bench = _bench_summary()
    md = _render_latency_histogram(bench, [])
    assert "(no runs)" in md


def test_latency_histogram_all_zero_elapsed() -> None:
    bench = _bench_summary(
        _bench_run("tiny/exploding_grad/0", fix_passes=True, bug_fails=True, elapsed=0.0),
    )
    md = _render_latency_histogram(bench, [])
    assert "zero elapsed" in md


# ---- cost breakdown ----


def test_cost_breakdown_lists_all_arms() -> None:
    baselines = [
        _baseline_summary("groq"),
        _baseline_summary("cerebras"),
    ]
    md = _render_cost_breakdown(baselines)
    assert "$0.00" in md
    assert "lensml-full" in md
    assert "groq" in md
    assert "cerebras" in md
    assert "free tier" in md


def test_cost_breakdown_no_baselines_just_lensml() -> None:
    md = _render_cost_breakdown([])
    assert "lensml-full" in md
    assert "$0.00" in md


# ---- end-to-end through render_markdown ----


def test_full_report_includes_all_plan_sections(tmp_path: Path) -> None:
    bench = _bench_summary(
        _bench_run("tiny/exploding_grad/0", fix_passes=True, bug_fails=True, elapsed=1.0),
        _bench_run("tiny/dead_relu/0", fix_passes=False, bug_fails=True, elapsed=1.5),
    )
    baselines = [
        _baseline_summary(
            "groq",
            _baseline_result(actual="exploding_grad", predicted="exploding_grad", fix_passes=True),
            _baseline_result(actual="dead_relu", predicted="exploding_grad", fix_passes=False),
        ),
    ]
    md = render_markdown(bench, baselines=baselines)

    # Plan §6.5 required sections.
    assert "Headline" in md
    assert "lensml-full vs LLM-only baselines" in md
    assert "Confusion matrix" in md
    assert "Latency histogram" in md
    assert "Cost breakdown" in md
    assert "Per-pathology" in md
    assert "Per-fixture" in md
    assert "What doesn't work yet" in md
    # The killer slide.
    assert "$0.00" in md


def test_build_report_persists_baselines_with_results(tmp_path: Path) -> None:
    """Round-trip: build_report → write → read JSON has the baseline
    results so ``lensml bench report`` can re-render later."""
    bench = _bench_summary(
        _bench_run("tiny/exploding_grad/0", fix_passes=True, bug_fails=True),
    )
    b = _baseline_summary(
        "groq",
        _baseline_result(actual="exploding_grad", predicted="exploding_grad", fix_passes=True),
    )
    report = build_report(bench, baselines=[b])
    md_path, json_path = report.write(tmp_path / "out")
    blob = json.loads(json_path.read_text(encoding="utf-8"))
    assert blob["baselines"][0]["provider"] == "groq"
    assert blob["baselines"][0]["results"][0]["diagnosed_class"] == "exploding_grad"
    assert "Confusion matrix" in md_path.read_text(encoding="utf-8")
