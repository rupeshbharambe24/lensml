"""Unit tests for lensml.replay.scenarios — the 3 canned demo paths."""

from __future__ import annotations

from lensml.detectors.dead_relu import DeadReluDetector
from lensml.detectors.exploding_grad import ExplodingGradDetector
from lensml.detectors.missing_normalize import MissingNormalizeDetector
from lensml.replay.scenarios import (
    SCENARIOS,
    generate,
    generate_dead_relu,
    generate_exploding_grad,
    generate_missing_normalize,
    list_scenarios,
)


def test_list_scenarios_has_three_entries() -> None:
    assert sorted(list_scenarios()) == sorted(["exploding_grad", "dead_relu", "missing_normalize"])


def test_registry_keys_match_helper() -> None:
    assert set(SCENARIOS.keys()) == set(list_scenarios())


def test_generate_unknown_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        generate("does_not_exist")


# ---- shape checks ----


def _structure_check(events: list[dict]) -> None:
    assert events[0]["kind"] == "run_started"
    assert events[-1]["kind"] == "run_ended"
    # Every event has run_id and ts.
    rid = events[0]["run_id"]
    for ev in events:
        assert ev["run_id"] == rid
        assert "ts" in ev
    # Timestamps are non-decreasing.
    from itertools import pairwise

    timestamps = [ev["ts"] for ev in events]
    for a, b in pairwise(timestamps):
        assert b >= a, f"ts went backwards: {a} -> {b}"


def test_exploding_grad_structure() -> None:
    events = generate_exploding_grad(pre_steps=20, post_steps=10)
    _structure_check(events)
    anomalies = [e for e in events if e["kind"] == "anomaly"]
    assert len(anomalies) == 1
    assert anomalies[0]["detector"] == "exploding_grad"


def test_dead_relu_structure() -> None:
    events = generate_dead_relu(pre_steps=20, post_steps=10)
    _structure_check(events)
    anomalies = [e for e in events if e["kind"] == "anomaly"]
    assert len(anomalies) == 1
    assert anomalies[0]["detector"] == "dead_relu"


def test_missing_normalize_structure() -> None:
    events = generate_missing_normalize(pre_steps=3, post_steps=10)
    _structure_check(events)
    anomalies = [e for e in events if e["kind"] == "anomaly"]
    assert len(anomalies) == 1
    assert anomalies[0]["detector"] == "missing_normalize"


# ---- determinism ----


def test_same_seed_same_output() -> None:
    a = generate_exploding_grad(seed=42)
    b = generate_exploding_grad(seed=42)
    # ts is wall-clock when base_ts=0 — drop it for the equality check.
    a2 = [{k: v for k, v in e.items() if k != "ts"} for e in a]
    b2 = [{k: v for k, v in e.items() if k != "ts"} for e in b]
    assert a2 == b2


def test_different_seeds_diverge() -> None:
    a = generate_dead_relu(seed=1)
    b = generate_dead_relu(seed=999)
    a_losses = [e.get("loss") for e in a if e["kind"] == "metric"]
    b_losses = [e.get("loss") for e in b if e["kind"] == "metric"]
    assert a_losses != b_losses


# ---- detector round-trip ----


def test_exploding_grad_actually_fires_detector() -> None:
    """Feeding the scenario events through the real detector should
    produce a detected anomaly — proves the scenario data is realistic."""
    events = generate_exploding_grad(pre_steps=30, post_steps=5)
    det = ExplodingGradDetector(warmup=10, window=15, multiplier=10.0, persist_steps=3)
    fired_count = 0
    for ev in events:
        if ev["kind"] != "metric":
            continue
        if det.observe(ev) is not None:
            fired_count += 1
    assert fired_count >= 1


def test_dead_relu_actually_fires_detector() -> None:
    events = generate_dead_relu(pre_steps=20, post_steps=5)
    det = DeadReluDetector(threshold=0.5, persist_steps=3)
    fired_count = 0
    for ev in events:
        if ev["kind"] != "metric":
            continue
        if det.observe(ev) is not None:
            fired_count += 1
    assert fired_count >= 1


def test_missing_normalize_actually_fires_detector() -> None:
    events = generate_missing_normalize()
    det = MissingNormalizeDetector()
    fired_count = 0
    for ev in events:
        if ev["kind"] != "metric":
            continue
        if det.observe(ev) is not None:
            fired_count += 1
    assert fired_count == 1  # single-shot


def test_metrics_have_required_telemetry_fields() -> None:
    """Every metric event has the fields the dashboard expects."""
    for fn in (generate_exploding_grad, generate_dead_relu, generate_missing_normalize):
        events = fn()
        metrics = [e for e in events if e["kind"] == "metric"]
        for ev in metrics:
            assert "step" in ev
            assert "loss" in ev
            assert "grad_l2" in ev
            assert "lr" in ev
