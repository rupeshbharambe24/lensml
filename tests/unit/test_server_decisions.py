"""Unit tests for lensml.server.decisions."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from lensml.server.decisions import (
    apply_decision,
    compute_diff_preview,
    parse_decision_payload,
)
from lensml.server.git import GitRepo

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git CLI not available")

_TRAINING_LOOP = (
    "def train_step(model, optimizer, loss):\n"
    "    optimizer.zero_grad()\n"
    "    loss.backward()\n"
    "    optimizer.step()\n"
)


# ---- parse_decision_payload ----


def test_parse_payload_valid_json() -> None:
    s = json.dumps({"template": "X", "params": {"a": 1}, "rationale": "r"})
    out = parse_decision_payload(s)
    assert out["template"] == "X"
    assert out["params"] == {"a": 1}


def test_parse_payload_invalid_json_returns_empty() -> None:
    assert parse_decision_payload("not json") == {}


def test_parse_payload_none_returns_empty() -> None:
    assert parse_decision_payload(None) == {}


def test_parse_payload_non_dict_returns_empty() -> None:
    assert parse_decision_payload(json.dumps([1, 2, 3])) == {}


# ---- compute_diff_preview ----


def _decision(template: str, params: dict, *, decision_id: int = 1, rationale: str = "") -> dict:
    return {
        "id": decision_id,
        "run_id": "r",
        "fix_template": template,
        "diff_text": json.dumps({"template": template, "params": params, "rationale": rationale}),
    }


def test_compute_preview_applicable(tmp_path: Path) -> None:
    script = tmp_path / "train.py"
    script.write_text(_TRAINING_LOOP, encoding="utf-8")
    decision = _decision("AddGradientClipping", {"max_norm": 1.0, "model_name": "model"})
    preview = compute_diff_preview(decision, script)

    assert preview.applicable is True
    assert preview.template == "AddGradientClipping"
    assert preview.new_source is not None
    assert "clip_grad_norm_" in preview.new_source
    assert preview.diff is not None
    assert "+" in preview.diff


def test_compute_preview_not_applicable_no_pattern(tmp_path: Path) -> None:
    script = tmp_path / "train.py"
    script.write_text("x = 1\n", encoding="utf-8")
    decision = _decision("AddGradientClipping", {})
    preview = compute_diff_preview(decision, script)
    assert preview.applicable is False
    assert preview.reason


def test_compute_preview_unknown_template(tmp_path: Path) -> None:
    script = tmp_path / "train.py"
    script.write_text(_TRAINING_LOOP, encoding="utf-8")
    decision = _decision("DoesNotExist", {})
    preview = compute_diff_preview(decision, script)
    assert preview.applicable is False
    assert "unknown fix template" in (preview.reason or "")


def test_compute_preview_missing_script(tmp_path: Path) -> None:
    decision = _decision("AddGradientClipping", {})
    preview = compute_diff_preview(decision, tmp_path / "missing.py")
    assert preview.applicable is False
    assert "script not found" in (preview.reason or "")


# ---- apply_decision ----


def test_apply_decision_writes_file_and_commits(tmp_path: Path) -> None:
    repo = GitRepo(tmp_path / "shadow")
    repo.init_if_needed()
    script = repo.work_dir / "train.py"
    script.write_text(_TRAINING_LOOP, encoding="utf-8")
    repo.commit_file(script, "seed train.py")

    decision = _decision(
        "AddGradientClipping",
        {"max_norm": 1.0, "model_name": "model"},
    )
    outcome = apply_decision(decision, script_path=script, repo=repo, run_id="r")

    assert outcome.applied
    assert outcome.commit
    assert outcome.diff is not None

    # File on disk now contains the patched source
    assert "clip_grad_norm_" in script.read_text(encoding="utf-8")

    # Commit message follows plan §3.3 convention
    log = repo.log_oneline(5)
    assert any("AddGradientClipping" in line and "decision 1" in line for line in log)


def test_apply_decision_returns_not_applied_when_pattern_missing(tmp_path: Path) -> None:
    repo = GitRepo(tmp_path / "shadow")
    repo.init_if_needed()
    script = repo.work_dir / "train.py"
    script.write_text("x = 1\n", encoding="utf-8")
    repo.commit_file(script, "seed")

    decision = _decision("AddGradientClipping", {})
    outcome = apply_decision(decision, script_path=script, repo=repo, run_id="r")
    assert not outcome.applied
    assert outcome.reason
