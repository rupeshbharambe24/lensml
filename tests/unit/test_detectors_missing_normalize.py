"""Unit tests for MissingNormalizeDetector (plan §4 row 4)."""

from __future__ import annotations

import math

from lensml.detectors.missing_normalize import MissingNormalizeDetector


def _det(**overrides: float) -> MissingNormalizeDetector:
    return MissingNormalizeDetector(**overrides)  # type: ignore[arg-type]


def _metric(step: int, **fields: object) -> dict[str, object]:
    return {"kind": "metric", "run_id": "r", "step": step, **fields}


def test_ignores_non_metric_events() -> None:
    det = _det()
    bad = {"kind": "anomaly", "input_stats": {"mean": 100.0, "min": 0.0, "max": 255.0}}
    assert det.observe(bad) is None


def test_returns_none_when_no_input_stats() -> None:
    det = _det()
    assert det.observe(_metric(0)) is None
    assert det.observe(_metric(1, input_stats="not a dict")) is None


def test_returns_none_for_normalized_imagenet_stats() -> None:
    det = _det()
    # Typical ImageNet-normalized image: mean near 0, std near 1
    stats = {"mean": 0.05, "std": 1.02, "min": -2.5, "max": 2.6}
    assert det.observe(_metric(0, input_stats=stats)) is None


def test_returns_none_for_zero_one_normalized() -> None:
    det = _det()
    # Image scaled to [0, 1] but not standardized — mean ~0.5, max=1.
    # Mean within ±3, max within max_bound, max <= 1.5 → should not fire.
    stats = {"mean": 0.48, "std": 0.25, "min": 0.0, "max": 1.0}
    assert det.observe(_metric(0, input_stats=stats)) is None


def test_fires_on_unsigned_uint8_stats() -> None:
    det = _det()
    # Raw uint8 image data: mean ≈ 127, max = 255
    stats = {"mean": 127.5, "std": 60.0, "min": 0.0, "max": 255.0}
    anomaly = det.observe(_metric(0, input_stats=stats))
    assert anomaly is not None
    assert anomaly.detector == "missing_normalize"
    assert anomaly.signal == "unnormalized_input"
    assert anomaly.evidence["input_mean"] == 127.5
    assert "mean_out_of_range" in anomaly.evidence["triggers"]
    assert "max_out_of_range" in anomaly.evidence["triggers"]
    assert "unsigned_range" in anomaly.evidence["triggers"]


def test_fires_on_mean_out_of_range_only() -> None:
    det = _det(mean_bound=3.0)
    stats = {"mean": 5.0, "std": 1.0, "min": -1.0, "max": 1.5}
    anomaly = det.observe(_metric(0, input_stats=stats))
    assert anomaly is not None
    assert "mean_out_of_range" in anomaly.evidence["triggers"]


def test_fires_only_once_per_run() -> None:
    det = _det()
    bad = {"mean": 127.5, "min": 0.0, "max": 255.0}
    first = det.observe(_metric(0, input_stats=bad))
    assert first is not None
    # Even after many more bad batches, no re-fire
    for i in range(1, 20):
        assert det.observe(_metric(i, input_stats=bad)) is None


def test_invalid_mean_value_ignored() -> None:
    det = _det()
    assert det.observe(_metric(0, input_stats={"mean": "x"})) is None
    assert det.observe(_metric(1, input_stats={"mean": math.nan})) is None
    assert det.observe(_metric(2, input_stats={"mean": math.inf})) is None


def test_zero_one_data_with_high_max_fires_unsigned_range() -> None:
    """Image data on [0, 2] (e.g. /128 instead of /255) is suspicious — both
    min ≥ 0 AND max > 1.5."""
    det = _det()
    stats = {"mean": 1.0, "std": 0.5, "min": 0.0, "max": 2.0}
    anomaly = det.observe(_metric(0, input_stats=stats))
    assert anomaly is not None
    assert "unsigned_range" in anomaly.evidence["triggers"]


def test_anomaly_to_event_round_trips() -> None:
    det = _det()
    stats = {"mean": 127.5, "min": 0.0, "max": 255.0}
    anomaly = det.observe(_metric(0, input_stats=stats))
    assert anomaly is not None
    payload = anomaly.to_event()
    assert payload["kind"] == "anomaly"
    assert payload["detector"] == "missing_normalize"
    assert payload["signal"] == "unnormalized_input"
