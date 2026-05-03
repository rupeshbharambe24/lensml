"""Benchmark report generator (plan §6.5, §8 D26).

Produces ``report.md`` + ``report.json`` from a ``BenchSummary`` and
optional baseline arms.

Sections rendered:
    - Headline (overall pass, fix-pass, bug-fails counts)
    - lensml-full vs LLM-only baselines (when baselines present)
    - Confusion matrix per baseline (predicted × actual pathology class)
    - Latency histogram (ASCII bars, lensml + each baseline arm)
    - Cost breakdown ($0 — the killer slide)
    - Per-pathology aggregate
    - Per-fixture pass/fail table
    - What doesn't work yet (fixture-by-fixture failure reasons)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lensml.bench.baseline import BaselineSummary
    from lensml.bench.runner import BenchSummary, FixtureRun


@dataclass
class Report:
    summary_dict: dict
    markdown: str
    generated_ts: float = field(default_factory=time.time)

    def write(self, dir_path: Path) -> tuple[Path, Path]:
        """Write report.md + report.json into ``dir_path``. Returns paths."""
        dir_path.mkdir(parents=True, exist_ok=True)
        md_path = dir_path / "report.md"
        json_path = dir_path / "report.json"
        md_path.write_text(self.markdown, encoding="utf-8")
        json_path.write_text(
            json.dumps(self.summary_dict, default=str, indent=2),
            encoding="utf-8",
        )
        return md_path, json_path


def _render_per_fixture_table(runs: list[FixtureRun]) -> str:
    if not runs:
        return "_(no fixtures)_\n"
    lines = [
        "| fixture | pathology | env | bug fails | fix passes | overall | elapsed |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in runs:
        lines.append(
            f"| `{r.fixture_slug}` | {r.pathology_class} | {r.host_env} | "
            f"{'✓' if r.bug_correctly_fails else '✗'} | "
            f"{'✓' if r.fix_passes else '✗'} | "
            f"{'✓' if r.overall_pass else '✗'} | "
            f"{r.elapsed:.1f}s |"
        )
    return "\n".join(lines) + "\n"


def _render_per_pathology(runs: list[FixtureRun]) -> str:
    by_class: dict[str, list[FixtureRun]] = {}
    for r in runs:
        by_class.setdefault(r.pathology_class, []).append(r)
    if not by_class:
        return "_(no fixtures)_\n"
    lines = [
        "| pathology | n | overall pass | fix-pass rate |",
        "| --- | --- | --- | --- |",
    ]
    for cls in sorted(by_class):
        bucket = by_class[cls]
        n = len(bucket)
        ok = sum(1 for r in bucket if r.overall_pass)
        fix_ok = sum(1 for r in bucket if r.fix_passes)
        lines.append(f"| {cls} | {n} | {ok}/{n} | {fix_ok}/{n} |")
    return "\n".join(lines) + "\n"


def _render_failures(runs: list[FixtureRun]) -> str:
    failures = [r for r in runs if not r.overall_pass]
    if not failures:
        return "_All fixtures pass cleanly._\n"
    lines = []
    for r in failures:
        bug_reason = r.bug.reason or "(passed all checks — bug isn't a bug?)"
        patched_reason = r.patched.reason or "(passed all checks but classified failed)"
        lines.append(f"- **`{r.fixture_slug}`** ({r.pathology_class})")
        if not r.bug_correctly_fails:
            lines.append(f"  - bug.py unexpectedly passed grader: {bug_reason}")
        if not r.fix_passes:
            lines.append(f"  - patched.py failed grader: {patched_reason}")
    return "\n".join(lines) + "\n"


def _confusion_matrix(baseline: BaselineSummary) -> dict[str, dict[str, int]]:
    """Build a {actual: {predicted: count}} confusion matrix from a
    baseline summary. ``predicted=None`` is bucketed under ``"(unparsed)"``."""
    matrix: dict[str, dict[str, int]] = {}
    for r in baseline.results:
        actual = r.pathology_class
        predicted = r.diagnosed_class or "(unparsed)"
        matrix.setdefault(actual, {})
        matrix[actual][predicted] = matrix[actual].get(predicted, 0) + 1
    return matrix


def _render_confusion_matrix(baseline: BaselineSummary) -> str:
    """Confusion matrix table per plan §6.5."""
    if not baseline.results:
        return "_(no baseline runs)_\n"
    matrix = _confusion_matrix(baseline)
    actuals = sorted(matrix.keys())
    predicted_set: set[str] = set()
    for row in matrix.values():
        predicted_set.update(row.keys())
    predicted = sorted(predicted_set)

    header = "| actual ↓ / predicted → | " + " | ".join(predicted) + " | total |"
    sep = "| --- |" + " --- |" * (len(predicted) + 1)
    lines = [header, sep]
    for actual in actuals:
        row = matrix[actual]
        cells = [str(row.get(p, 0)) for p in predicted]
        total = sum(row.values())
        # Highlight the diagonal cell (correct diagnosis count) when present.
        if actual in predicted:
            idx = predicted.index(actual)
            cells[idx] = f"**{cells[idx]}**" if cells[idx] != "0" else cells[idx]
        lines.append(f"| {actual} | " + " | ".join(cells) + f" | {total} |")
    return "\n".join(lines) + "\n"


def _render_latency_histogram(bench: BenchSummary, baselines: list[BaselineSummary]) -> str:
    """ASCII histogram of per-fixture elapsed times."""
    series: list[tuple[str, list[float]]] = []
    if bench.runs:
        series.append(("lensml-full", [r.elapsed for r in bench.runs]))
    for b in baselines:
        if b.results:
            series.append((f"baseline·{b.provider}", [r.elapsed for r in b.results]))
    if not series:
        return "_(no runs)_\n"

    # Define shared bucket edges across all series so bars are comparable.
    all_values = [v for _, vals in series for v in vals]
    if not all_values:
        return "_(no timing data)_\n"
    max_v = max(all_values)
    if max_v <= 0:
        return "_(all runs reported zero elapsed)_\n"
    n_buckets = 6
    width = max_v / n_buckets
    edges = [width * i for i in range(n_buckets + 1)]

    lines = ["```"]
    for name, vals in series:
        lines.append(f"{name}:")
        for i in range(n_buckets):
            lo, hi = edges[i], edges[i + 1]
            count = sum(1 for v in vals if lo <= v < hi or (i == n_buckets - 1 and v == hi))
            bar = "#" * count
            lines.append(f"  [{lo:5.1f}–{hi:5.1f}s] {bar} ({count})")
    lines.append("```")
    return "\n".join(lines) + "\n"


def _render_cost_breakdown(baselines: list[BaselineSummary]) -> str:
    """Plan §6.5 — the $0 cost slide is the killer line."""
    arms = ["lensml-full"] + [f"baseline · {b.provider}" for b in baselines]
    lines = [
        "| arm | tier | cost |",
        "| --- | --- | --- |",
    ]
    for a in arms:
        lines.append(f"| {a} | free | **$0.00** |")
    lines.append("")
    lines.append(
        "_Every provider used (Groq, Cerebras, Gemini, OpenRouter, Mistral) "
        "is on its free tier. The whole 30-day build, including baselines, "
        "ran on $0._"
    )
    return "\n".join(lines) + "\n"


def _render_baseline_comparison(bench: BenchSummary, baselines: list[BaselineSummary]) -> str:
    pct = lambda x: f"{x * 100:.1f}%"  # noqa: E731
    lines = [
        "| arm | n | diagnosis acc | fix-pass rate | overall pass |",
        "| --- | --- | --- | --- | --- |",
        f"| **lensml-full** | {bench.total} | n/a (unique) | "
        f"{pct(bench.fix_pass_rate)} | {pct(bench.overall_pass_rate)} |",
    ]
    for b in baselines:
        lines.append(
            f"| llm-only · {b.provider} | {b.total} | "
            f"{pct(b.diagnosis_accuracy)} | {pct(b.fix_pass_rate)} | "
            f"{b.fix_passes}/{b.total} |"
        )
    return "\n".join(lines) + "\n"


def render_markdown(
    summary: BenchSummary,
    *,
    title: str = "lensml-bench report",
    baselines: list[BaselineSummary] | None = None,
) -> str:
    pct = lambda x: f"{x * 100:.1f}%"  # noqa: E731
    out: list[str] = []
    out.append(f"# {title}\n")
    out.append(
        f"_{time.strftime('%Y-%m-%d %H:%M:%S')} — "
        f"{summary.total} fixtures, {summary.elapsed:.1f}s_\n"
    )
    out.append("")
    out.append("## Headline\n")
    out.append(
        f"- **Overall pass:** {summary.overall_pass}/{summary.total}  "
        f"({pct(summary.overall_pass_rate)})\n"
        f"- **Fix passes grader:** {summary.fix_pass}/{summary.total}  "
        f"({pct(summary.fix_pass_rate)})\n"
        f"- **Bug correctly fails:** {summary.bug_correctly_fails}/{summary.total}\n"
    )
    out.append("")
    if baselines:
        out.append("## lensml-full vs LLM-only baselines\n")
        out.append(
            "Plan §6.3: the LLM-only baseline is the credibility lynchpin. "
            "If lensml-full doesn't beat single-shot LLM diagnosis, the "
            "integration story doesn't hold up. Numbers below are over the "
            "same fixture set.\n"
        )
        out.append(_render_baseline_comparison(summary, baselines))
        out.append("")

        out.append("## Confusion matrix (LLM-only baseline diagnoses)\n")
        out.append(
            "Each row is a fixture's *actual* pathology; each column is the "
            "single-shot LLM diagnosis. Diagonal cells (correct) are bolded.\n"
        )
        for b in baselines:
            out.append(f"### baseline · {b.provider}\n")
            out.append(_render_confusion_matrix(b))
            out.append("")

    out.append("## Latency histogram\n")
    out.append(_render_latency_histogram(summary, baselines or []))
    out.append("")

    out.append("## Cost breakdown\n")
    out.append(_render_cost_breakdown(baselines or []))
    out.append("")

    out.append("## Per-pathology\n")
    out.append(_render_per_pathology(summary.runs))
    out.append("")
    out.append("## Per-fixture\n")
    out.append(_render_per_fixture_table(summary.runs))
    out.append("")
    out.append("## What doesn't work yet\n")
    out.append(_render_failures(summary.runs))
    return "\n".join(out)


def build_report(
    summary: BenchSummary,
    *,
    title: str = "lensml-bench report",
    baselines: list[BaselineSummary] | None = None,
) -> Report:
    summary_dict = summary.to_dict()
    if baselines:
        summary_dict["baselines"] = [b.to_dict() for b in baselines]
    return Report(
        summary_dict=summary_dict,
        markdown=render_markdown(summary, title=title, baselines=baselines),
    )


__all__ = ["Report", "build_report", "render_markdown"]
