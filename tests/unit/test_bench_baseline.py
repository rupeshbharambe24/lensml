"""Unit tests for lensml.bench.baseline."""

from __future__ import annotations

import difflib
import json
from pathlib import Path

from lensml.bench.baseline import (
    BaselineResult,
    BaselineRunner,
    BaselineSummary,
    parse_diagnosis_class,
    parse_diff_from_response,
)
from lensml.bench.fixtures import discover_fixtures


# ---- helpers ----


def _bug_source(loss: float) -> str:
    return f'import json\nprint(json.dumps({{"final_loss": {loss}}}))\n'


def _make_scratch_fixture(
    base: Path,
    *,
    pathology: str = "exploding_grad",
    bug_loss: float = 1.5,
    loss_threshold: float = 0.5,
) -> None:
    p = base / "tiny" / pathology / "0"
    p.mkdir(parents=True, exist_ok=True)
    bug = _bug_source(bug_loss)
    fixed = _bug_source(0.1)
    diff = "".join(
        difflib.unified_diff(
            bug.splitlines(keepends=True),
            fixed.splitlines(keepends=True),
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


def _canned_response(pathology_keyword: str, bug_loss: float, fixed_loss: float) -> str:
    """Synthesize a plausible LLM response for the scratch fixtures."""
    bug = _bug_source(bug_loss)
    fixed = _bug_source(fixed_loss)
    diff = "".join(
        difflib.unified_diff(
            bug.splitlines(keepends=True),
            fixed.splitlines(keepends=True),
            fromfile="a/bug.py",
            tofile="b/bug.py",
        )
    )
    return f"Diagnosis: {pathology_keyword}.\n\n```diff\n{diff}```\n"


# ---- parse_diagnosis_class ----


def test_parse_diagnosis_known_synonyms() -> None:
    assert parse_diagnosis_class("Diagnosis: exploding gradient.") == "exploding_grad"
    assert parse_diagnosis_class("Diagnosis: dead ReLU on layer 3.") == "dead_relu"
    assert parse_diagnosis_class("Diagnosis: missing normalization.") == "missing_normalize"
    assert parse_diagnosis_class("Diagnosis: lr too high → loss oscillates.") == "lr_too_high"
    assert parse_diagnosis_class("Diagnosis: bad initialization (kaiming needed)") == "bad_init"


def test_parse_diagnosis_returns_none_for_unknown() -> None:
    assert parse_diagnosis_class("Diagnosis: I don't know.") is None
    assert parse_diagnosis_class("") is None


def test_parse_diagnosis_case_insensitive() -> None:
    assert parse_diagnosis_class("DIAGNOSIS: EXPLODING GRADIENTS detected") == "exploding_grad"


# ---- parse_diff_from_response ----


def test_parse_diff_from_fenced_block() -> None:
    text = "Some prose.\n\n```diff\n--- a/x\n+++ b/x\n@@\n-a\n+b\n```\nDone."
    out = parse_diff_from_response(text)
    assert out.startswith("--- a/x")
    assert "@@" in out


def test_parse_diff_supports_patch_tag() -> None:
    text = "```patch\n--- a/x\n+++ b/x\n@@\n-a\n+b\n```"
    assert "--- a/x" in parse_diff_from_response(text)


def test_parse_diff_returns_empty_when_no_block() -> None:
    assert parse_diff_from_response("just prose, no diff") == ""


# ---- BaselineRunner with mocked complete ----


def _runner_for(tmp_path: Path, complete) -> BaselineRunner:
    return BaselineRunner(
        provider="mock",
        complete=complete,
        fixtures=discover_fixtures(tmp_path),
    )


def test_run_one_records_correct_diagnosis_and_passes_grade(tmp_path: Path) -> None:
    _make_scratch_fixture(tmp_path)
    canned = _canned_response("exploding gradient", bug_loss=1.5, fixed_loss=0.1)
    result = _runner_for(tmp_path, lambda _p: canned).run_one(discover_fixtures(tmp_path)[0])
    assert isinstance(result, BaselineResult)
    assert result.diagnosed_class == "exploding_grad"
    assert result.diagnosis_correct is True
    assert result.diff_applied is True
    assert result.fix_passes is True
    assert result.error is None


def test_run_one_handles_no_diff(tmp_path: Path) -> None:
    _make_scratch_fixture(tmp_path)
    response = "Diagnosis: exploding gradient.\nNo diff produced — sorry."
    result = _runner_for(tmp_path, lambda _p: response).run_one(discover_fixtures(tmp_path)[0])
    assert result.diagnosed_class == "exploding_grad"
    assert result.diff_applied is False
    assert result.fix_passes is False
    assert "did not return a diff" in (result.patched.reason or "")


def test_run_one_handles_unappliable_diff(tmp_path: Path) -> None:
    _make_scratch_fixture(tmp_path)
    bad_diff = "```diff\n--- a/x\n+++ b/x\n@@ -1,1 +1,1 @@\n-totally\n+different\n```"
    response = f"Diagnosis: dead ReLU.\n\n{bad_diff}"
    result = _runner_for(tmp_path, lambda _p: response).run_one(discover_fixtures(tmp_path)[0])
    assert result.diff_applied is False
    assert "did not apply" in (result.patched.reason or "")


def test_run_one_records_provider_error(tmp_path: Path) -> None:
    _make_scratch_fixture(tmp_path)

    def _boom(_prompt: str) -> str:
        raise RuntimeError("provider unavailable")

    result = _runner_for(tmp_path, _boom).run_one(discover_fixtures(tmp_path)[0])
    assert result.error is not None
    assert "RuntimeError" in result.error
    assert result.fix_passes is False


def test_run_one_wrong_diagnosis_still_runs_diff(tmp_path: Path) -> None:
    """If the LLM mis-diagnoses but the diff happens to apply + grade, the
    fix_passes flag should still flip True. This is exactly the kind of
    "lucky guess" pattern lensml-full should beat."""
    _make_scratch_fixture(tmp_path)
    canned = _canned_response("dead ReLU", bug_loss=1.5, fixed_loss=0.1)
    result = _runner_for(tmp_path, lambda _p: canned).run_one(discover_fixtures(tmp_path)[0])
    assert result.diagnosed_class == "dead_relu"
    assert result.diagnosis_correct is False  # ground truth was exploding_grad
    assert result.fix_passes is True


# ---- run_all + BaselineSummary ----


def test_run_all_aggregates(tmp_path: Path) -> None:
    _make_scratch_fixture(tmp_path, pathology="exploding_grad")
    _make_scratch_fixture(tmp_path, pathology="dead_relu")
    canned = _canned_response("exploding gradient", bug_loss=1.5, fixed_loss=0.1)
    summary = _runner_for(tmp_path, lambda _p: canned).run_all()
    assert isinstance(summary, BaselineSummary)
    assert summary.total == 2
    # Both diffs apply, both grades pass.
    assert summary.fix_passes == 2
    # But only one fixture's pathology is "exploding_grad".
    assert summary.diagnosis_correct == 1
    assert summary.diagnosis_accuracy == 0.5
    assert summary.fix_pass_rate == 1.0


def test_run_all_filter_by_env(tmp_path: Path) -> None:
    _make_scratch_fixture(tmp_path, pathology="exploding_grad")
    runner = _runner_for(tmp_path, lambda _p: "")
    s_tiny = runner.run_all(env="tiny")
    s_small = runner.run_all(env="small")
    assert s_tiny.total == 1
    assert s_small.total == 0


def test_summary_zero_total_safe(tmp_path: Path) -> None:
    summary = _runner_for(tmp_path, lambda _p: "").run_all()
    assert summary.total == 0
    assert summary.diagnosis_accuracy == 0.0
    assert summary.fix_pass_rate == 0.0


def test_summary_to_dict_includes_provider() -> None:
    s = BaselineSummary(provider="cerebras", total=2, fix_passes=1, diagnosis_correct=0)
    d = s.to_dict()
    assert d["provider"] == "cerebras"
    assert d["total"] == 2
    assert d["fix_pass_rate"] == 0.5
