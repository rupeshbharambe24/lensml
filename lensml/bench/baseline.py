"""LLM-only baseline runner (plan §6.3 — the credibility lynchpin).

For every fixture, a *baseline* is the answer you get from a single-shot
LLM prompt without lensml's hooks, detectors, or AST patcher::

    "Here is a broken training script. Diagnose the bug and propose a
    minimal unified diff that fixes it."

If lensml-full beats this baseline meaningfully — better diagnosis
accuracy, better pass@1 on the grader — the bench tells the story:
"lensml's value is the *integration*, not just calling an LLM."

This module is the baseline orchestration: take a callable
``complete(prompt) -> str`` that wraps any provider, run it against
every fixture, parse the LLM's diff out of the response, grade the
patched script, return a ``BaselineSummary`` shaped like
``BenchSummary`` so reports can compare them side by side.
"""

from __future__ import annotations

import logging
import re
import shutil
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

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

DEFAULT_PROMPT_TEMPLATE = """\
You are an ML engineer reviewing a broken PyTorch training script. The
script either crashes, fails to converge, or otherwise produces a
worse-than-expected result. Diagnose the bug and propose a minimal
fix as a unified diff.

Output format:
    1. One short paragraph naming the bug (e.g. "exploding gradients",
       "dead ReLU", "missing input normalization", "learning rate too
       high"). Start the paragraph with `Diagnosis:`.
    2. The unified diff inside a fenced code block tagged ```diff …```.
       The diff should rewrite ``a/bug.py`` → ``b/bug.py`` and apply
       cleanly with `patch -p1`.

Notes:
    - Keep the diff small. Do NOT rewrite the whole file.
    - Match existing indentation exactly.
    - Do not add new imports unless absolutely required.

Broken script (`bug.py`):
```python
{bug_source}
```

Pathology hint (one of: exploding_grad, lr_too_high, dead_relu,
missing_normalize, model_eval_not_called, bad_init, vanishing_grad,
dataloader_bottleneck): pick whichever best fits and name it in the
diagnosis paragraph."""


# Synonyms map LLM-output prose → canonical detector slugs.
_DIAGNOSIS_SYNONYMS: dict[str, str] = {
    "exploding gradient": "exploding_grad",
    "exploding gradients": "exploding_grad",
    "gradient explosion": "exploding_grad",
    "grad explod": "exploding_grad",
    "learning rate too high": "lr_too_high",
    "lr too high": "lr_too_high",
    "high learning rate": "lr_too_high",
    "loss oscillat": "lr_too_high",
    "dead relu": "dead_relu",
    "dying relu": "dead_relu",
    "dead activation": "dead_relu",
    "missing normaliz": "missing_normalize",
    "no normalization": "missing_normalize",
    "unnormalized": "missing_normalize",
    "model.eval": "model_eval_not_called",
    "eval mode": "model_eval_not_called",
    "evaluation mode": "model_eval_not_called",
    "bad initialization": "bad_init",
    "weight init": "bad_init",
    "kaiming": "bad_init",
    "vanishing grad": "vanishing_grad",
    "saturating activation": "vanishing_grad",
    "dataloader bottleneck": "dataloader_bottleneck",
    "num_workers": "dataloader_bottleneck",
    "lr schedule": "lr_schedule_mismatch",
    "scheduler": "lr_schedule_mismatch",
}


@dataclass
class BaselineResult:
    """Per-fixture outcome of an LLM-only baseline."""

    fixture_slug: str
    pathology_class: str
    host_env: str
    diagnosed_class: str | None  # canonical slug, or None if unparseable
    diagnosis_correct: bool
    proposed_diff: str  # may be empty if the LLM didn't produce one
    diff_applied: bool
    patched: GradeResult
    raw_response: str
    elapsed: float = 0.0
    error: str | None = None

    @property
    def fix_passes(self) -> bool:
        return self.patched.passed

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture_slug": self.fixture_slug,
            "pathology_class": self.pathology_class,
            "host_env": self.host_env,
            "diagnosed_class": self.diagnosed_class,
            "diagnosis_correct": self.diagnosis_correct,
            "diff_applied": self.diff_applied,
            "fix_passes": self.fix_passes,
            "patched": self.patched.to_dict(),
            "elapsed": self.elapsed,
            "error": self.error,
        }


@dataclass
class BaselineSummary:
    """Aggregate of a baseline run over a fixture set."""

    provider: str
    total: int = 0
    diagnosis_correct: int = 0
    fix_passes: int = 0
    diff_applied: int = 0
    elapsed: float = 0.0
    results: list[BaselineResult] = field(default_factory=list)

    @property
    def diagnosis_accuracy(self) -> float:
        return self.diagnosis_correct / self.total if self.total else 0.0

    @property
    def fix_pass_rate(self) -> float:
        return self.fix_passes / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "total": self.total,
            "diagnosis_correct": self.diagnosis_correct,
            "fix_passes": self.fix_passes,
            "diff_applied": self.diff_applied,
            "diagnosis_accuracy": self.diagnosis_accuracy,
            "fix_pass_rate": self.fix_pass_rate,
            "elapsed": self.elapsed,
            "results": [r.to_dict() for r in self.results],
        }


_DIFF_FENCE_RE = re.compile(r"```(?:diff|patch)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_DIAGNOSIS_LINE_RE = re.compile(
    r"diagnosis\s*[:\-]\s*(.+?)(?:\n\n|\n```|\Z)", re.IGNORECASE | re.DOTALL
)


def parse_diff_from_response(response: str) -> str:
    """Pull the first ```diff …``` (or ```patch``` or untagged) block.

    Returns "" if no fenced block exists.
    """
    m = _DIFF_FENCE_RE.search(response)
    if m is None:
        return ""
    return m.group(1)


def parse_diagnosis_class(response: str) -> str | None:
    """Map the LLM's prose diagnosis to a canonical detector slug.

    Looks at the line after ``Diagnosis:`` (case-insensitive) plus the
    full response as a fallback. Returns ``None`` if no synonym matches.
    """
    haystacks: list[str] = []
    m = _DIAGNOSIS_LINE_RE.search(response)
    if m is not None:
        haystacks.append(m.group(1).lower())
    haystacks.append(response.lower())
    for hay in haystacks:
        for needle, slug in _DIAGNOSIS_SYNONYMS.items():
            if needle in hay:
                return slug
    return None


CompleteFn = Callable[[str], str]
AsyncCompleteFn = Callable[[str], Any]  # returns awaitable


class BaselineRunner:
    """Run an LLM-only baseline across a fixture set."""

    def __init__(
        self,
        *,
        provider: str,
        complete: CompleteFn,
        fixtures: list[Fixture] | None = None,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        """`complete(prompt) -> response` is a sync callable. Tests pass
        a mock; production wiring wraps a ``LensmlLLM.complete`` chain."""
        self.provider = provider
        self.complete = complete
        self.fixtures = fixtures if fixtures is not None else discover_fixtures()
        self.prompt_template = prompt_template
        self.timeout = timeout

    # ---- single fixture ----

    def run_one(self, fixture: Fixture) -> BaselineResult:
        prompt = self.prompt_template.format(bug_source=fixture.bug_source)
        t0 = time.time()
        raw = ""
        error: str | None = None
        try:
            raw = self.complete(prompt) or ""
        except Exception as e:  # any provider error
            error = f"{type(e).__name__}: {str(e)[:200]}"
            log.warning("baseline: complete() raised on %s: %s", fixture.slug, error)

        diagnosed = parse_diagnosis_class(raw)
        diagnosis_correct = diagnosed == fixture.meta.pathology_class
        diff = parse_diff_from_response(raw)

        diff_applied = False
        patched_result: GradeResult
        if not diff:
            patched_result = GradeResult(
                fixture_slug=fixture.slug,
                variant="baseline",
                checks=GradeChecks(),
                reason="LLM did not return a diff block",
            )
        else:
            try:
                patched_src = apply_diff(fixture.bug_source, diff)
                diff_applied = True
            except ValueError as e:
                patched_src = None
                patched_result = GradeResult(
                    fixture_slug=fixture.slug,
                    variant="baseline",
                    checks=GradeChecks(),
                    reason=f"baseline diff did not apply: {e}",
                )
            if diff_applied:
                with tempfile.TemporaryDirectory(prefix="lensml-baseline-") as tmp:
                    tmp_path = Path(tmp)
                    for fname in ("bug.py", "fix.diff", "grade.py", "meta.json"):
                        shutil.copy2(fixture.path / fname, tmp_path / fname)
                    patched_path = tmp_path / "patched.py"
                    patched_path.write_text(patched_src or "", encoding="utf-8")
                    patched_result = grade_script(
                        patched_path,
                        fixture,
                        variant="baseline",
                        timeout=self.timeout,
                    )

        return BaselineResult(
            fixture_slug=fixture.slug,
            pathology_class=fixture.meta.pathology_class,
            host_env=fixture.meta.host_env,
            diagnosed_class=diagnosed,
            diagnosis_correct=diagnosis_correct,
            proposed_diff=diff,
            diff_applied=diff_applied,
            patched=patched_result,
            raw_response=raw,
            elapsed=time.time() - t0,
            error=error,
        )

    # ---- top level ----

    def run_all(
        self,
        *,
        env: str | None = None,
        pathology: str | None = None,
    ) -> BaselineSummary:
        summary = BaselineSummary(provider=self.provider)
        t0 = time.time()
        for fix in self._filter(env=env, pathology=pathology):
            result = self.run_one(fix)
            summary.results.append(result)
            summary.total += 1
            if result.diagnosis_correct:
                summary.diagnosis_correct += 1
            if result.fix_passes:
                summary.fix_passes += 1
            if result.diff_applied:
                summary.diff_applied += 1
        summary.elapsed = time.time() - t0
        return summary

    def _filter(self, *, env: str | None, pathology: str | None) -> Iterable[Fixture]:
        for fix in self.fixtures:
            if env is not None and fix.meta.host_env != env:
                continue
            if pathology is not None and fix.meta.pathology_class != pathology:
                continue
            yield fix


__all__ = [
    "DEFAULT_PROMPT_TEMPLATE",
    "BaselineResult",
    "BaselineRunner",
    "BaselineSummary",
    "parse_diagnosis_class",
    "parse_diff_from_response",
]
