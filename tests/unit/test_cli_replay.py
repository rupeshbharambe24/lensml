"""Unit tests for the `lensml replay` and `lensml record` CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from lensml.cli import app
from lensml.replay import read_jsonl

runner = CliRunner()


def test_replay_dry_run_with_scenario() -> None:
    result = runner.invoke(app, ["replay", "--scenario", "exploding_grad", "--dry-run"])
    assert result.exit_code == 0
    assert "metric" in result.stdout
    assert "anomaly" in result.stdout
    assert "run_started" in result.stdout
    assert "run_ended" in result.stdout


def test_replay_dry_run_with_jsonl_file(tmp_path: Path) -> None:
    p = tmp_path / "in.jsonl"
    with p.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"kind": "metric", "step": 0}) + "\n")
        f.write(json.dumps({"kind": "anomaly", "step": 0}) + "\n")
    result = runner.invoke(app, ["replay", str(p), "--dry-run"])
    assert result.exit_code == 0
    assert "metric: 1" in result.stdout
    assert "anomaly: 1" in result.stdout


def test_replay_unknown_scenario_exits_nonzero() -> None:
    result = runner.invoke(app, ["replay", "--scenario", "nope", "--dry-run"])
    assert result.exit_code != 0


def test_replay_no_args_exits_nonzero() -> None:
    result = runner.invoke(app, ["replay"])
    assert result.exit_code != 0
    assert "scenario" in result.stdout.lower() or "scenario" in (result.stderr or "").lower()


def test_replay_missing_file_exits_nonzero(tmp_path: Path) -> None:
    result = runner.invoke(app, ["replay", str(tmp_path / "missing.jsonl"), "--dry-run"])
    assert result.exit_code != 0


def test_record_writes_jsonl(tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl"
    result = runner.invoke(app, ["record", str(out), "--scenario", "missing_normalize"])
    assert result.exit_code == 0, result.stdout
    assert out.exists()
    events = read_jsonl(out)
    assert events[0]["kind"] == "run_started"
    assert events[-1]["kind"] == "run_ended"
    detectors = [e["detector"] for e in events if e["kind"] == "anomaly"]
    assert "missing_normalize" in detectors


def test_record_run_id_override(tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl"
    result = runner.invoke(
        app,
        ["record", str(out), "--scenario", "dead_relu", "--run-id", "demo-replay"],
    )
    assert result.exit_code == 0
    events = read_jsonl(out)
    assert {e["run_id"] for e in events} == {"demo-replay"}


def test_record_unknown_scenario_exits_nonzero(tmp_path: Path) -> None:
    result = runner.invoke(app, ["record", str(tmp_path / "out.jsonl"), "--scenario", "no_such"])
    assert result.exit_code != 0
