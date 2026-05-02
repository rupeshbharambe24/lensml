"""Unit tests for LrTooHighDetector (plan §4 row 2)."""

from __future__ import annotations

import math

from lensml.detectors.lr_too_high import LrTooHighDetector


def _det(**overrides: int | float) -> LrTooHighDetector:
    kwargs: dict[str, float | int] = {"window": 10, "flip_ratio": 0.6, "cooldown": 5}
    kwargs.update(overrides)
    return LrTooHighDetector(**kwargs)  # type: ignore[arg-type]


def _metric(step: int, **fields: object) -> dict[str, object]:
    return {"kind": "metric", "run_id": "r", "step": step, **fields}


def test_ignores_non_metric_events() -> None:
    det = _det()
    assert det.observe({"kind": "anomaly", "loss": 1.0}) is None


def test_returns_none_when_no_loss_field() -> None:
    det = _det()
    for i in range(20):
        assert det.observe(_metric(i)) is None


def test_invalid_loss_values_ignored() -> None:
    det = _det()
    assert det.observe(_metric(0, loss="not a number")) is None
    assert det.observe(_metric(1, loss=math.nan)) is None
    assert det.observe(_metric(2, loss=math.inf)) is None


def test_returns_none_during_warmup() -> None:
    det = _det(window=10)
    # Even with extreme oscillation, can't fire until the window is full.
    losses = [1.0, 2.0, 1.0, 2.0]
    for i, lv in enumerate(losses):
        assert det.observe(_metric(i, loss=lv)) is None


def test_smooth_decreasing_loss_does_not_fire() -> None:
    det = _det(window=10)
    for i in range(50):
        loss = 5.0 / (1.0 + i * 0.1)  # smooth monotonic decrease
        assert det.observe(_metric(i, loss=loss)) is None


def test_oscillating_loss_fires() -> None:
    det = _det(window=10, flip_ratio=0.6, cooldown=0)
    # Strict alternation: every step is a sign-flip.
    losses = [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
    anomaly = None
    for i, lv in enumerate(losses):
        anomaly = det.observe(_metric(i, loss=lv, lr=0.1))
    assert anomaly is not None
    assert anomaly.detector == "lr_too_high"
    assert anomaly.signal == "loss_oscillation"
    assert anomaly.severity == "warning"
    assert anomaly.evidence["flip_ratio"] >= 0.6
    assert anomaly.evidence["window"] == 10
    assert anomaly.evidence["lr"] == 0.1


def test_partial_oscillation_below_threshold_does_not_fire() -> None:
    det = _det(window=10, flip_ratio=0.6)
    # Only 2 sign-flips out of 8 possible — well under the threshold.
    losses = [5.0, 4.0, 4.5, 4.0, 3.5, 3.0, 3.5, 3.0, 2.5, 2.0]
    for i, lv in enumerate(losses):
        assert det.observe(_metric(i, loss=lv)) is None


def test_cooldown_prevents_immediate_refire() -> None:
    det = _det(window=10, flip_ratio=0.6, cooldown=20)
    losses = [1.0, 2.0] * 5
    fired = None
    for i, lv in enumerate(losses):
        out = det.observe(_metric(i, loss=lv))
        if out is not None:
            fired = out
            fire_step = i
            break
    assert fired is not None
    # Continue feeding oscillation — should not fire during cooldown.
    for i in range(fire_step + 1, fire_step + 1 + 15):
        assert det.observe(_metric(i, loss=1.0 if i % 2 == 0 else 2.0)) is None


def test_lr_field_optional() -> None:
    det = _det(window=10, flip_ratio=0.6, cooldown=0)
    losses = [1.0, 2.0] * 5
    anomaly = None
    for i, lv in enumerate(losses):
        anomaly = det.observe(_metric(i, loss=lv))
    assert anomaly is not None
    assert anomaly.evidence["lr"] is None


def test_anomaly_to_event_round_trips() -> None:
    det = _det(window=10, flip_ratio=0.6, cooldown=0)
    losses = [1.0, 2.0] * 5
    anomaly = None
    for i, lv in enumerate(losses):
        anomaly = det.observe(_metric(i, loss=lv))
    assert anomaly is not None
    payload = anomaly.to_event()
    assert payload["kind"] == "anomaly"
    assert payload["detector"] == "lr_too_high"
    assert payload["signal"] == "loss_oscillation"
