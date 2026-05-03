"""Unit test for `lensml bench report` — stitches bench + baseline JSONs."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from lensml.cli import app

runner = CliRunner()


def _write_bench_json(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "total": 2,
                "fix_pass": 2,
                "bug_correctly_fails": 2,
                "overall_pass": 2,
                "elapsed": 5.0,
                "runs": [
                    {
                        "fixture_slug": "tiny/exploding_grad/0",
                        "pathology_class": "exploding_grad",
                        "host_env": "tiny",
                        "elapsed": 2.5,
                        "fix_passes": True,
                        "bug_correctly_fails": True,
                        "overall_pass": True,
                        "bug": {
                            "fixture_slug": "tiny/exploding_grad/0",
                            "variant": "bug",
                            "checks": {
                                "runs": True,
                                "loss_converged": False,
                                "metric_passed": None,
                            },
                        },
                        "patched": {
                            "fixture_slug": "tiny/exploding_grad/0",
                            "variant": "patched",
                            "checks": {"runs": True, "loss_converged": True, "metric_passed": None},
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_baseline_json(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "provider": "groq",
                "total": 1,
                "diagnosis_correct": 1,
                "fix_passes": 1,
                "diff_applied": 1,
                "elapsed": 3.0,
                "results": [
                    {
                        "fixture_slug": "tiny/exploding_grad/0",
                        "pathology_class": "exploding_grad",
                        "host_env": "tiny",
                        "diagnosed_class": "exploding_grad",
                        "diagnosis_correct": True,
                        "diff_applied": True,
                        "fix_passes": True,
                        "elapsed": 3.0,
                        "patched": {
                            "fixture_slug": "tiny/exploding_grad/0",
                            "variant": "baseline",
                            "checks": {"runs": True, "loss_converged": True, "metric_passed": None},
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_bench_report_stitches_bench_and_baseline(tmp_path: Path) -> None:
    bench_path = tmp_path / "bench.json"
    baseline_path = tmp_path / "baseline-groq.json"
    out_dir = tmp_path / "out"
    _write_bench_json(bench_path)
    _write_baseline_json(baseline_path)

    result = runner.invoke(
        app,
        [
            "bench",
            "report",
            "--bench",
            str(bench_path),
            "--baseline",
            str(baseline_path),
            "--out",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert (out_dir / "report.md").exists()
    assert (out_dir / "report.json").exists()
    md = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "Confusion matrix" in md
    assert "Latency histogram" in md
    assert "$0.00" in md
    assert "groq" in md


def test_bench_report_works_without_baselines(tmp_path: Path) -> None:
    bench_path = tmp_path / "bench.json"
    out_dir = tmp_path / "out"
    _write_bench_json(bench_path)
    result = runner.invoke(
        app,
        ["bench", "report", "--bench", str(bench_path), "--out", str(out_dir)],
    )
    assert result.exit_code == 0
    md = (out_dir / "report.md").read_text(encoding="utf-8")
    # No comparison or confusion-matrix section without baselines.
    assert "lensml-full vs LLM-only baselines" not in md
    # Latency + cost still rendered.
    assert "Latency histogram" in md
    assert "$0.00" in md
