"""Unit tests for BadInitDetector (plan §4 row 7)."""

from __future__ import annotations

import math

from lensml.detectors.bad_init import BadInitDetector


def _det(**overrides: float | int) -> BadInitDetector:
    kwargs: dict[str, float | int] = {"multiplier": 2.0, "max_step": 5}
    kwargs.update(overrides)
    return BadInitDetector(**kwargs)  # type: ignore[arg-type]


def _metric(step: int, **fields: object) -> dict[str, object]:
    return {"kind": "metric", "run_id": "r", "step": step, **fields}


def test_ignores_non_metric_events() -> None:
    det = _det()
    assert det.observe({"kind": "anomaly", "loss": 100.0, "num_classes": 10}) is None


def test_returns_none_when_no_num_classes() -> None:
    det = _det()
    assert det.observe(_metric(0, loss=100.0)) is None


def test_returns_none_when_loss_within_expected() -> None:
    """A 10-class problem should expect CE ≈ log(10) ≈ 2.30. 2.0 is fine."""
    det = _det(multiplier=2.0)
    assert det.observe(_metric(0, loss=2.0, num_classes=10)) is None


def test_fires_when_loss_above_threshold() -> None:
    """log(10) ≈ 2.30; 2× threshold = 4.6. A loss of 5.0 should fire."""
    det = _det(multiplier=2.0)
    anomaly = det.observe(_metric(0, loss=5.0, num_classes=10))
    assert anomaly is not None
    assert anomaly.detector == "bad_init"
    assert anomaly.signal == "loss_at_init_high"
    assert anomaly.evidence["loss"] == 5.0
    assert anomaly.evidence["num_classes"] == 10
    assert anomaly.evidence["expected_log_k"] == math.log(10)


def test_fires_only_once_per_run() -> None:
    det = _det(multiplier=2.0, max_step=5)
    first = det.observe(_metric(0, loss=10.0, num_classes=10))
    assert first is not None
    for i in range(1, 5):
        assert det.observe(_metric(i, loss=10.0, num_classes=10)) is None


def test_skips_after_max_step_window() -> None:
    """Detector only checks the very first few steps."""
    det = _det(max_step=3)
    # No firing on step > max_step even with bad loss
    assert det.observe(_metric(10, loss=100.0, num_classes=10)) is None
    # And once we've seen a step past the window, even a step <= max_step
    # in a later event won't fire (window-locked)
    assert det.observe(_metric(0, loss=100.0, num_classes=10)) is None


def test_invalid_loss_or_num_classes_ignored() -> None:
    det = _det()
    assert det.observe(_metric(0, loss=math.nan, num_classes=10)) is None
    assert det.observe(_metric(0, loss=math.inf, num_classes=10)) is None
    assert det.observe(_metric(0, loss="x", num_classes=10)) is None
    assert det.observe(_metric(0, loss=10.0, num_classes=1)) is None  # K must be >=2


def test_anomaly_to_event_round_trips() -> None:
    det = _det()
    anomaly = det.observe(_metric(0, loss=10.0, num_classes=10))
    assert anomaly is not None
    payload = anomaly.to_event()
    assert payload["kind"] == "anomaly"
    assert payload["detector"] == "bad_init"
    assert payload["signal"] == "loss_at_init_high"
