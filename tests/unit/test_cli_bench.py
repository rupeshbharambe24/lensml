"""Unit tests for the `lensml bench` CLI subcommands."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from lensml.cli import app

runner = CliRunner()


def test_bench_list_packages_packaged_fixtures() -> None:
    result = runner.invoke(app, ["bench", "list"])
    assert result.exit_code == 0
    assert "tiny/exploding_grad/0" in result.stdout
    assert "tiny/dead_relu/0" in result.stdout


def test_bench_list_filters_by_env() -> None:
    result = runner.invoke(app, ["bench", "list", "--env", "tiny"])
    assert result.exit_code == 0
    # all packaged fixtures are tiny right now → output is non-empty
    assert "tiny/" in result.stdout


def test_bench_list_unknown_env_returns_empty() -> None:
    result = runner.invoke(app, ["bench", "list", "--env", "huge"])
    assert result.exit_code == 0
    assert "(no fixtures match)" in result.stdout


def test_bench_run_with_pathology_filter_writes_report(tmp_path: Path) -> None:
    """Use a non-existent pathology to keep the run trivial — empty
    summary still produces a valid report."""
    out_dir = tmp_path / "results"
    result = runner.invoke(
        app,
        [
            "bench",
            "run",
            "--pathology",
            "lr_schedule_mismatch_xxx",  # filters out everything
            "--out",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0
    assert "0/0 overall pass" in result.stdout
    assert (out_dir / "report.md").exists()
    assert (out_dir / "report.json").exists()
