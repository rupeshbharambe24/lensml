"""Unit tests for VanishingGradDetector (plan §4 row 8)."""

from __future__ import annotations

import math

from lensml.detectors.vanishing_grad import VanishingGradDetector


def _det(**overrides: float | int) -> VanishingGradDetector:
    kwargs: dict[str, float | int] = {
        "floor": 1e-7,
        "ratio_threshold": 100.0,
        "persist_steps": 3,
    }
    kwargs.update(overrides)
    return VanishingGradDetector(**kwargs)  # type: ignore[arg-type]


def _metric(step: int, **fields: object) -> dict[str, object]:
    return {"kind": "metric", "run_id": "r", "step": step, **fields}


def test_ignores_non_metric_events() -> None:
    det = _det()
    bad = {"kind": "anomaly", "per_layer_grad_l2": {"l1": 1.0, "l2": 1e-9}}
    assert det.observe(bad) is None


def test_returns_none_when_no_per_layer_grad() -> None:
    det = _det()
    assert det.observe(_metric(0)) is None


def test_returns_none_when_only_one_layer() -> None:
    det = _det()
    assert det.observe(_metric(0, per_layer_grad_l2={"only": 1e-9})) is None


def test_returns_none_when_min_above_floor() -> None:
    det = _det(floor=1e-7, persist_steps=2)
    grads = {"l1": 0.5, "l2": 1e-3}  # min way above floor
    for i in range(5):
        assert det.observe(_metric(i, per_layer_grad_l2=grads)) is None


def test_returns_none_when_ratio_below_threshold() -> None:
    """Min < floor but max/min ratio is small — both layers vanishing,
    not a depth-related vanishing-gradient case."""
    det = _det(floor=1e-7, ratio_threshold=100.0, persist_steps=2)
    grads = {"l1": 5e-9, "l2": 1e-9}  # ratio = 5
    for i in range(5):
        assert det.observe(_metric(i, per_layer_grad_l2=grads)) is None


def test_persist_steps_fires() -> None:
    det = _det(floor=1e-7, ratio_threshold=100.0, persist_steps=3)
    grads = {"shallow": 1.0, "mid": 0.1, "deep": 1e-9}
    assert det.observe(_metric(0, per_layer_grad_l2=grads)) is None
    assert det.observe(_metric(1, per_layer_grad_l2=grads)) is None
    anomaly = det.observe(_metric(2, per_layer_grad_l2=grads))
    assert anomaly is not None
    assert anomaly.detector == "vanishing_grad"
    assert anomaly.signal == "grad_vanishing"
    assert anomaly.evidence["min_layer"] == "deep"
    assert anomaly.evidence["max_layer"] == "shallow"
    assert anomaly.evidence["ratio"] >= 100.0
    assert anomaly.evidence["persist_steps"] == 3


def test_streak_resets_on_healthy_step() -> None:
    det = _det(persist_steps=3)
    bad = {"a": 1.0, "b": 1e-9}
    good = {"a": 1.0, "b": 0.5}
    det.observe(_metric(0, per_layer_grad_l2=bad))
    det.observe(_metric(1, per_layer_grad_l2=bad))
    det.observe(_metric(2, per_layer_grad_l2=good))  # reset
    assert det.observe(_metric(3, per_layer_grad_l2=bad)) is None
    assert det.observe(_metric(4, per_layer_grad_l2=bad)) is None


def test_streak_resets_after_firing() -> None:
    det = _det(persist_steps=3)
    bad = {"a": 1.0, "b": 1e-9}
    det.observe(_metric(0, per_layer_grad_l2=bad))
    det.observe(_metric(1, per_layer_grad_l2=bad))
    first = det.observe(_metric(2, per_layer_grad_l2=bad))
    assert first is not None
    assert det.observe(_metric(3, per_layer_grad_l2=bad)) is None
    assert det.observe(_metric(4, per_layer_grad_l2=bad)) is None


def test_invalid_grads_filtered() -> None:
    det = _det(persist_steps=2)
    grads = {"a": 1.0, "b": math.nan, "c": 1e-9, "d": "x"}
    det.observe(_metric(0, per_layer_grad_l2=grads))
    anomaly = det.observe(_metric(1, per_layer_grad_l2=grads))
    assert anomaly is not None
    assert anomaly.evidence["min_layer"] == "c"


def test_anomaly_to_event_round_trips() -> None:
    det = _det(persist_steps=2)
    grads = {"a": 1.0, "b": 1e-9}
    det.observe(_metric(0, per_layer_grad_l2=grads))
    anomaly = det.observe(_metric(1, per_layer_grad_l2=grads))
    assert anomaly is not None
    payload = anomaly.to_event()
    assert payload["kind"] == "anomaly"
    assert payload["detector"] == "vanishing_grad"
    assert payload["signal"] == "grad_vanishing"
