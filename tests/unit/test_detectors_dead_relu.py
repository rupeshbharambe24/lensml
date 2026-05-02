"""Unit tests for DeadReluDetector (plan §4 row 3)."""

from __future__ import annotations

from lensml.detectors.dead_relu import DeadReluDetector


def _det(**overrides: int | float) -> DeadReluDetector:
    kwargs: dict[str, float | int] = {"threshold": 0.5, "persist_steps": 3}
    kwargs.update(overrides)
    return DeadReluDetector(**kwargs)  # type: ignore[arg-type]


def _activations(**dead_fractions: float) -> dict[str, dict[str, float]]:
    return {name: {"mean": 0.0, "dead_fraction": v} for name, v in dead_fractions.items()}


def _metric(step: int, **fields: object) -> dict[str, object]:
    return {"kind": "metric", "run_id": "r", "step": step, **fields}


def test_ignores_non_metric_events() -> None:
    det = _det()
    assert det.observe({"kind": "anomaly", "activations": _activations(relu1=0.99)}) is None


def test_returns_none_when_no_activations_field() -> None:
    det = _det()
    assert det.observe(_metric(0)) is None
    assert det.observe(_metric(1, activations="not a dict")) is None
    assert det.observe(_metric(2, activations={})) is None


def test_returns_none_below_threshold() -> None:
    det = _det()
    for i in range(10):
        ev = _metric(i, activations=_activations(relu1=0.3, relu2=0.4))
        assert det.observe(ev) is None


def test_one_bad_step_does_not_fire() -> None:
    det = _det(persist_steps=3)
    assert det.observe(_metric(0, activations=_activations(relu1=0.99))) is None


def test_persist_steps_fires_on_third() -> None:
    det = _det(persist_steps=3)
    assert det.observe(_metric(0, activations=_activations(relu1=0.99))) is None
    assert det.observe(_metric(1, activations=_activations(relu1=0.99))) is None
    anomaly = det.observe(_metric(2, activations=_activations(relu1=0.99)))
    assert anomaly is not None
    assert anomaly.detector == "dead_relu"
    assert anomaly.signal == "dead_fraction_high"
    assert anomaly.severity == "warning"
    assert anomaly.run_id == "r"
    assert anomaly.step == 2
    assert anomaly.evidence["module"] == "relu1"
    assert anomaly.evidence["dead_fraction"] == 0.99
    assert anomaly.evidence["persist_steps"] == 3


def test_streak_resets_on_healthy_step() -> None:
    det = _det(persist_steps=3)
    det.observe(_metric(0, activations=_activations(relu1=0.99)))
    det.observe(_metric(1, activations=_activations(relu1=0.99)))
    # Healthy step in the middle resets the streak.
    det.observe(_metric(2, activations=_activations(relu1=0.1)))
    # Two more bad steps shouldn't fire — need three again.
    assert det.observe(_metric(3, activations=_activations(relu1=0.99))) is None
    assert det.observe(_metric(4, activations=_activations(relu1=0.99))) is None


def test_streak_resets_after_firing() -> None:
    det = _det(persist_steps=3)
    det.observe(_metric(0, activations=_activations(relu1=0.99)))
    det.observe(_metric(1, activations=_activations(relu1=0.99)))
    first = det.observe(_metric(2, activations=_activations(relu1=0.99)))
    assert first is not None
    assert det.observe(_metric(3, activations=_activations(relu1=0.99))) is None
    assert det.observe(_metric(4, activations=_activations(relu1=0.99))) is None


def test_picks_worst_module_when_multiple_dead() -> None:
    det = _det(persist_steps=2)
    det.observe(_metric(0, activations=_activations(relu1=0.6, relu2=0.92, relu3=0.55)))
    anomaly = det.observe(_metric(1, activations=_activations(relu1=0.6, relu2=0.92, relu3=0.55)))
    assert anomaly is not None
    assert anomaly.evidence["module"] == "relu2"
    assert anomaly.evidence["dead_fraction"] == 0.92


def test_malformed_activation_entries_ignored() -> None:
    det = _det(persist_steps=2)
    # Stat dict missing 'dead_fraction' / non-dict values are skipped.
    bad = {"relu1": {"mean": 0.0}, "relu2": "junk", "relu3": {"dead_fraction": 0.99}}
    det.observe(_metric(0, activations=bad))
    anomaly = det.observe(_metric(1, activations=bad))
    assert anomaly is not None
    assert anomaly.evidence["module"] == "relu3"


def test_anomaly_to_event_round_trips() -> None:
    det = _det(persist_steps=2)
    det.observe(_metric(0, activations=_activations(relu1=0.99)))
    anomaly = det.observe(_metric(1, activations=_activations(relu1=0.99)))
    assert anomaly is not None
    payload = anomaly.to_event()
    assert payload["kind"] == "anomaly"
    assert payload["detector"] == "dead_relu"
    assert payload["signal"] == "dead_fraction_high"
    assert payload["evidence"]["module"] == "relu1"
