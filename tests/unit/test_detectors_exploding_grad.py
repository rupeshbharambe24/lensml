"""Unit tests for ExplodingGradDetector (plan §4 row 1)."""

from __future__ import annotations

import math

from lensml.detectors.exploding_grad import ExplodingGradDetector


def _det(**overrides: int | float) -> ExplodingGradDetector:
    """Test-friendly detector with tight defaults."""
    kwargs: dict[str, float | int] = {
        "warmup": 5,
        "window": 10,
        "multiplier": 10.0,
        "persist_steps": 3,
    }
    kwargs.update(overrides)
    return ExplodingGradDetector(**kwargs)  # type: ignore[arg-type]


def _metric(step: int, **fields: object) -> dict[str, object]:
    return {"kind": "metric", "run_id": "r", "step": step, **fields}


# ---- baseline behavior ----


def test_ignores_non_metric_events() -> None:
    det = _det()
    assert det.observe({"kind": "anomaly", "grad_l2": 1000.0}) is None
    assert det.observe({"kind": "run_started"}) is None


def test_returns_none_during_warmup() -> None:
    det = _det(warmup=5)
    for i in range(5):
        assert det.observe(_metric(i, grad_l2=1.0)) is None


def test_returns_none_for_normal_grad_after_warmup() -> None:
    det = _det()
    for i in range(15):
        det.observe(_metric(i, grad_l2=1.0))
    # Slight variation around baseline — should not fire
    assert det.observe(_metric(15, grad_l2=1.5)) is None
    assert det.observe(_metric(16, grad_l2=0.7)) is None


def test_missing_grad_l2_returns_none() -> None:
    det = _det()
    for i in range(10):
        det.observe(_metric(i, grad_l2=1.0))
    assert det.observe(_metric(10)) is None
    assert det.observe(_metric(11, grad_l2="not a number")) is None


# ---- spike detection ----


def test_one_spike_does_not_fire() -> None:
    det = _det()
    for i in range(10):
        det.observe(_metric(i, grad_l2=1.0))
    assert det.observe(_metric(10, grad_l2=50.0)) is None


def test_two_spikes_do_not_fire() -> None:
    det = _det()
    for i in range(10):
        det.observe(_metric(i, grad_l2=1.0))
    assert det.observe(_metric(10, grad_l2=50.0)) is None
    assert det.observe(_metric(11, grad_l2=50.0)) is None


def test_three_consecutive_spikes_fire() -> None:
    det = _det()
    for i in range(10):
        det.observe(_metric(i, grad_l2=1.0))
    det.observe(_metric(10, grad_l2=50.0))
    det.observe(_metric(11, grad_l2=50.0))
    anomaly = det.observe(_metric(12, grad_l2=50.0))
    assert anomaly is not None
    assert anomaly.detector == "exploding_grad"
    assert anomaly.signal == "grad_spike"
    assert anomaly.severity == "critical"
    assert anomaly.run_id == "r"
    assert anomaly.step == 12
    assert anomaly.evidence["grad_l2"] == 50.0
    assert anomaly.evidence["ratio"] >= 10.0
    assert anomaly.evidence["persist_steps"] == 3


def test_streak_resets_on_normal_value() -> None:
    det = _det()
    for i in range(10):
        det.observe(_metric(i, grad_l2=1.0))
    det.observe(_metric(10, grad_l2=50.0))  # spike 1
    det.observe(_metric(11, grad_l2=50.0))  # spike 2
    det.observe(_metric(12, grad_l2=1.0))  # reset
    # Now next two spikes should not fire — streak rebuilds from zero
    assert det.observe(_metric(13, grad_l2=50.0)) is None
    assert det.observe(_metric(14, grad_l2=50.0)) is None


def test_streak_resets_after_firing() -> None:
    det = _det()
    for i in range(10):
        det.observe(_metric(i, grad_l2=1.0))
    det.observe(_metric(10, grad_l2=50.0))
    det.observe(_metric(11, grad_l2=50.0))
    first = det.observe(_metric(12, grad_l2=50.0))
    assert first is not None
    # Next two spikes should NOT fire — streak was reset to 0
    assert det.observe(_metric(13, grad_l2=50.0)) is None
    assert det.observe(_metric(14, grad_l2=50.0)) is None
    # Third spike re-fires
    second = det.observe(_metric(15, grad_l2=50.0))
    assert second is not None


# ---- nan / inf shortcuts ----


def test_nan_loss_fires_immediately_no_warmup() -> None:
    det = _det()
    anomaly = det.observe(_metric(0, loss=math.nan))
    assert anomaly is not None
    assert anomaly.signal == "nan_loss"
    assert anomaly.evidence["loss"] == "NaN"


def test_inf_grad_fires_immediately() -> None:
    det = _det()
    anomaly = det.observe(_metric(0, grad_l2=math.inf))
    assert anomaly is not None
    assert anomaly.signal == "inf_grad"
    assert anomaly.severity == "critical"


def test_finite_loss_does_not_fire_via_nan_path() -> None:
    det = _det()
    for i in range(10):
        det.observe(_metric(i, grad_l2=1.0, loss=0.5))
    # Loss is finite — only grad_l2 path could fire
    assert det.observe(_metric(10, grad_l2=1.0, loss=0.4)) is None


# ---- robustness ----


def test_zero_baseline_does_not_fire_on_tiny_value() -> None:
    """If history is all zeros, threshold floor (1e-6) prevents false positives
    on any nonzero gradient."""
    det = _det()
    for i in range(10):
        det.observe(_metric(i, grad_l2=0.0))
    # Tiny nonzero — should not fire on first observation alone
    assert det.observe(_metric(10, grad_l2=0.0001)) is None


def test_anomaly_to_event_round_trips() -> None:
    """Anomaly.to_event() produces the wire-shape expected by the broker."""
    det = _det()
    for i in range(10):
        det.observe(_metric(i, grad_l2=1.0))
    det.observe(_metric(10, grad_l2=100.0))
    det.observe(_metric(11, grad_l2=100.0))
    anomaly = det.observe(_metric(12, grad_l2=100.0))
    assert anomaly is not None
    payload = anomaly.to_event()
    assert payload["kind"] == "anomaly"
    assert payload["detector"] == "exploding_grad"
    assert payload["run_id"] == "r"
    assert payload["step"] == 12
    assert payload["signal"] == "grad_spike"
    assert payload["severity"] == "critical"
    assert "evidence" in payload
    assert "ts" in payload
