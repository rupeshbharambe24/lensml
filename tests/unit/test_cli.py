"""Smoke tests for the lensml CLI."""

from __future__ import annotations

from typer.testing import CliRunner

from lensml.cli import app

runner = CliRunner()


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    out = result.output.lower()
    assert "dashboard" in out
    assert "watch" in out
    assert "bench" in out
    assert "replay" in out


def test_dashboard_help_describes_options() -> None:
    result = runner.invoke(app, ["dashboard", "--help"])
    assert result.exit_code == 0
    assert "host" in result.output.lower()
    assert "port" in result.output.lower()


def test_watch_emits_guidance_and_exits_nonzero() -> None:
    """Until v0.2, `lensml watch` prints a hint instead of running."""
    result = runner.invoke(app, ["watch", "train.py"])
    assert result.exit_code == 2
    assert "lensml.watch" in result.output


def test_bench_emits_guidance_and_exits_nonzero() -> None:
    result = runner.invoke(app, ["bench"])
    assert result.exit_code == 2


def test_replay_emits_guidance_and_exits_nonzero() -> None:
    result = runner.invoke(app, ["replay", "session.jsonl"])
    assert result.exit_code == 2
