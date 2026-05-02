"""Unit tests for ModelEvalNotCalledDetector (plan §4 row 6)."""

from __future__ import annotations

from lensml.detectors.model_eval_not_called import ModelEvalNotCalledDetector


def _det() -> ModelEvalNotCalledDetector:
    return ModelEvalNotCalledDetector()


def test_ignores_train_phase_events() -> None:
    det = _det()
    train_event = {"kind": "metric", "run_id": "r", "step": 0, "model_training": True}
    assert det.observe(train_event) is None


def test_ignores_validation_with_eval_mode() -> None:
    det = _det()
    val_event = {
        "kind": "metric",
        "run_id": "r",
        "step": 100,
        "phase": "val",
        "model_training": False,
    }
    assert det.observe(val_event) is None


def test_no_model_training_field_means_no_evidence() -> None:
    """Detector silent when the trainer doesn't publish the flag."""
    det = _det()
    val_event = {"kind": "metric", "run_id": "r", "step": 100, "phase": "val"}
    assert det.observe(val_event) is None


def test_fires_on_validation_in_train_mode_metric_phase() -> None:
    det = _det()
    val_event = {
        "kind": "metric",
        "run_id": "r",
        "step": 100,
        "phase": "val",
        "model_training": True,
    }
    anomaly = det.observe(val_event)
    assert anomaly is not None
    assert anomaly.detector == "model_eval_not_called"
    assert anomaly.signal == "val_in_train_mode"
    assert anomaly.severity == "warning"
    assert anomaly.run_id == "r"
    assert anomaly.step == 100
    assert anomaly.evidence["model_training"] is True
    assert anomaly.evidence["phase"] == "val"


def test_fires_on_validation_kind_in_train_mode() -> None:
    det = _det()
    val_event = {
        "kind": "validation",
        "run_id": "r",
        "step": 200,
        "model_training": True,
    }
    anomaly = det.observe(val_event)
    assert anomaly is not None
    assert anomaly.detector == "model_eval_not_called"


def test_records_offending_module_list() -> None:
    det = _det()
    val_event = {
        "kind": "metric",
        "phase": "val",
        "step": 100,
        "model_training": True,
        "training_modules": ["bn1", "dropout1"],
    }
    anomaly = det.observe(val_event)
    assert anomaly is not None
    assert anomaly.evidence["modules"] == ["bn1", "dropout1"]


def test_fires_only_once_per_run() -> None:
    det = _det()
    val_event = {
        "kind": "metric",
        "phase": "val",
        "step": 100,
        "model_training": True,
    }
    first = det.observe(val_event)
    assert first is not None
    # Subsequent val batches in the same wrong state should NOT re-fire.
    for step in range(101, 110):
        repeat = dict(val_event)
        repeat["step"] = step
        assert det.observe(repeat) is None


def test_non_bool_training_flag_ignored() -> None:
    det = _det()
    # 1 is truthy but not bool — should be ignored to avoid false positives
    # from sloppy serialization.
    val_event = {
        "kind": "metric",
        "phase": "val",
        "step": 1,
        "model_training": 1,
    }
    assert det.observe(val_event) is None


def test_anomaly_to_event_round_trips() -> None:
    det = _det()
    val_event = {
        "kind": "metric",
        "phase": "val",
        "step": 100,
        "model_training": True,
    }
    anomaly = det.observe(val_event)
    assert anomaly is not None
    payload = anomaly.to_event()
    assert payload["kind"] == "anomaly"
    assert payload["detector"] == "model_eval_not_called"
    assert payload["signal"] == "val_in_train_mode"
