"""Unit tests for DataloaderBottleneckDetector (plan §4 row 5)."""

from __future__ import annotations

from lensml.detectors.dataloader_bottleneck import DataloaderBottleneckDetector


def _det(**overrides: float | int) -> DataloaderBottleneckDetector:
    kwargs: dict[str, float | int] = {
        "ratio_threshold": 0.3,
        "num_workers_ratio_threshold": 0.15,
        "persist_steps": 3,
    }
    kwargs.update(overrides)
    return DataloaderBottleneckDetector(**kwargs)  # type: ignore[arg-type]


def _metric(step: int, **fields: object) -> dict[str, object]:
    return {"kind": "metric", "run_id": "r", "step": step, **fields}


def test_ignores_non_metric_events() -> None:
    det = _det()
    bad = {"kind": "anomaly", "dataloader_stats": {"batch_wait_ms": 100, "step_time_ms": 100}}
    assert det.observe(bad) is None


def test_returns_none_when_no_dataloader_stats() -> None:
    det = _det()
    assert det.observe(_metric(0)) is None
    assert det.observe(_metric(1, dataloader_stats="not a dict")) is None


def test_low_wait_ratio_does_not_fire() -> None:
    det = _det(ratio_threshold=0.3, persist_steps=3)
    for i in range(10):
        stats = {"batch_wait_ms": 5.0, "step_time_ms": 100.0, "num_workers": 4}
        assert det.observe(_metric(i, dataloader_stats=stats)) is None


def test_high_wait_ratio_fires_after_persist_steps() -> None:
    det = _det(ratio_threshold=0.3, persist_steps=3)
    stats = {"batch_wait_ms": 50.0, "step_time_ms": 100.0, "num_workers": 4}
    assert det.observe(_metric(0, dataloader_stats=stats)) is None
    assert det.observe(_metric(1, dataloader_stats=stats)) is None
    anomaly = det.observe(_metric(2, dataloader_stats=stats))
    assert anomaly is not None
    assert anomaly.detector == "dataloader_bottleneck"
    assert anomaly.signal == "batch_wait_high"
    assert anomaly.evidence["wait_ratio"] == 0.5
    assert anomaly.evidence["num_workers"] == 4
    assert anomaly.evidence["persist_steps"] == 3


def test_num_workers_zero_uses_softer_threshold() -> None:
    """A 20% wait ratio fires when num_workers=0 (soft threshold 0.15) but not
    when num_workers > 0 (hard threshold 0.3)."""
    soft = _det(ratio_threshold=0.3, num_workers_ratio_threshold=0.15, persist_steps=2)
    stats = {"batch_wait_ms": 20.0, "step_time_ms": 100.0, "num_workers": 0}
    soft.observe(_metric(0, dataloader_stats=stats))
    anomaly = soft.observe(_metric(1, dataloader_stats=stats))
    assert anomaly is not None

    hard = _det(ratio_threshold=0.3, num_workers_ratio_threshold=0.15, persist_steps=2)
    stats2 = {"batch_wait_ms": 20.0, "step_time_ms": 100.0, "num_workers": 8}
    hard.observe(_metric(0, dataloader_stats=stats2))
    assert hard.observe(_metric(1, dataloader_stats=stats2)) is None


def test_streak_resets_on_low_step() -> None:
    det = _det(ratio_threshold=0.3, persist_steps=3)
    bad = {"batch_wait_ms": 50.0, "step_time_ms": 100.0, "num_workers": 4}
    good = {"batch_wait_ms": 5.0, "step_time_ms": 100.0, "num_workers": 4}
    det.observe(_metric(0, dataloader_stats=bad))
    det.observe(_metric(1, dataloader_stats=bad))
    det.observe(_metric(2, dataloader_stats=good))
    assert det.observe(_metric(3, dataloader_stats=bad)) is None
    assert det.observe(_metric(4, dataloader_stats=bad)) is None


def test_fires_only_once_per_run() -> None:
    det = _det(persist_steps=2)
    bad = {"batch_wait_ms": 50.0, "step_time_ms": 100.0, "num_workers": 4}
    det.observe(_metric(0, dataloader_stats=bad))
    first = det.observe(_metric(1, dataloader_stats=bad))
    assert first is not None
    for i in range(2, 10):
        assert det.observe(_metric(i, dataloader_stats=bad)) is None


def test_invalid_step_time_ignored() -> None:
    det = _det()
    assert det.observe(_metric(0, dataloader_stats={"batch_wait_ms": 5, "step_time_ms": 0})) is None
    assert (
        det.observe(_metric(1, dataloader_stats={"batch_wait_ms": "x", "step_time_ms": 100}))
        is None
    )
