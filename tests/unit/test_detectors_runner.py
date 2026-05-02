"""Unit tests for DetectorRunner."""

from __future__ import annotations

import asyncio

import pytest

from lensml.detectors.base import Anomaly, Detector
from lensml.detectors.exploding_grad import ExplodingGradDetector
from lensml.detectors.runner import DetectorRunner
from lensml.server.sse import EventBroker
from lensml.server.store import EventStore


class _AlwaysFires(Detector):
    """Test detector: emits an anomaly for every metric event."""

    name = "always"

    def observe(self, event: dict) -> Anomaly | None:
        if event.get("kind") != "metric":
            return None
        return Anomaly(
            detector=self.name,
            run_id=event.get("run_id", "default"),
            step=event.get("step", 0),
            signal="test",
            evidence={"src": event},
        )


class _Crashes(Detector):
    """Test detector: raises on every observe — runner must isolate."""

    name = "crashes"

    def observe(self, event: dict) -> Anomaly | None:
        raise RuntimeError("boom")


async def _drain_until_anomaly(broker: EventBroker, timeout: float = 1.0) -> list[dict]:
    """Subscribe to the broker, return events until we see an anomaly or timeout."""
    sub = broker.subscribe()
    received: list[dict] = []
    try:
        while True:
            event = await asyncio.wait_for(sub.get(), timeout=timeout)
            received.append(event)
            if event.get("kind") == "anomaly":
                return received
    except TimeoutError:
        return received
    finally:
        broker.unsubscribe(sub)


@pytest.mark.asyncio
async def test_runner_lifecycle() -> None:
    broker = EventBroker()
    store = EventStore(":memory:")
    runner = DetectorRunner(broker, store, [_AlwaysFires()])

    await runner.start()
    assert runner._task is not None

    await runner.stop()
    assert runner._task is None
    store.close()


@pytest.mark.asyncio
async def test_runner_publishes_anomaly_to_broker() -> None:
    broker = EventBroker()
    store = EventStore(":memory:")
    runner = DetectorRunner(broker, store, [_AlwaysFires()])
    await runner.start()
    try:
        # Subscribe AFTER runner starts so we don't receive its own subscribe queue
        consumer_drain = asyncio.create_task(_drain_until_anomaly(broker))
        # Tiny yield so the consumer's subscribe registers
        await asyncio.sleep(0.01)
        broker.publish({"kind": "metric", "run_id": "r", "step": 1, "grad_l2": 1.0})
        events = await consumer_drain

        anomalies = [e for e in events if e.get("kind") == "anomaly"]
        assert len(anomalies) == 1
        assert anomalies[0]["detector"] == "always"
        assert anomalies[0]["step"] == 1
    finally:
        await runner.stop()
        store.close()


@pytest.mark.asyncio
async def test_runner_persists_anomaly_to_store() -> None:
    broker = EventBroker()
    store = EventStore(":memory:")
    runner = DetectorRunner(broker, store, [_AlwaysFires()])
    await runner.start()
    try:
        store.begin_run("r")
        broker.publish({"kind": "metric", "run_id": "r", "step": 5, "grad_l2": 1.0})
        # Yield enough times for the runner task to consume + persist
        for _ in range(20):
            await asyncio.sleep(0.01)
            if store.event_count("r") > 0:
                break

        anomalies = store.get_events("r", kind="anomaly")
        assert len(anomalies) == 1
        assert anomalies[0]["payload"]["detector"] == "always"
    finally:
        await runner.stop()
        store.close()


@pytest.mark.asyncio
async def test_runner_ignores_non_metric_events() -> None:
    """run_started, anomaly, run_ended events should not be re-processed."""
    broker = EventBroker()
    store = EventStore(":memory:")
    runner = DetectorRunner(broker, store, [_AlwaysFires()])
    await runner.start()
    try:
        store.begin_run("r")
        # Non-metric events
        broker.publish({"kind": "run_started", "run_id": "r"})
        broker.publish({"kind": "run_ended", "run_id": "r", "step": 0})
        broker.publish({"kind": "anomaly", "run_id": "r", "step": 0})
        for _ in range(10):
            await asyncio.sleep(0.01)
        # No anomaly stored — detector never observed any metric
        anomalies = store.get_events("r", kind="anomaly")
        assert len(anomalies) == 0
    finally:
        await runner.stop()
        store.close()


@pytest.mark.asyncio
async def test_runner_isolates_crashing_detector() -> None:
    """One crashing detector must not stop healthy detectors from firing."""
    broker = EventBroker()
    store = EventStore(":memory:")
    runner = DetectorRunner(broker, store, [_Crashes(), _AlwaysFires()])
    await runner.start()
    try:
        store.begin_run("r")
        broker.publish({"kind": "metric", "run_id": "r", "step": 0, "grad_l2": 1.0})
        for _ in range(20):
            await asyncio.sleep(0.01)
            if store.event_count("r") > 0:
                break

        anomalies = store.get_events("r", kind="anomaly")
        # _AlwaysFires fired once even though _Crashes raised
        assert len(anomalies) == 1
        assert anomalies[0]["payload"]["detector"] == "always"
    finally:
        await runner.stop()
        store.close()


@pytest.mark.asyncio
async def test_runner_with_real_exploding_grad_detector() -> None:
    """End-to-end: real detector + runner + a synthetic spike sequence."""
    broker = EventBroker()
    store = EventStore(":memory:")
    detector = ExplodingGradDetector(warmup=3, persist_steps=3, multiplier=10.0)
    runner = DetectorRunner(broker, store, [detector])
    await runner.start()
    try:
        store.begin_run("r")
        # Warmup with normal gradients
        for i in range(5):
            broker.publish({"kind": "metric", "run_id": "r", "step": i, "grad_l2": 1.0})
        # Three consecutive spikes — should fire
        for i in range(5, 8):
            broker.publish({"kind": "metric", "run_id": "r", "step": i, "grad_l2": 100.0})
        # Wait for runner to process
        for _ in range(40):
            await asyncio.sleep(0.01)
            if store.get_events("r", kind="anomaly"):
                break

        anomalies = store.get_events("r", kind="anomaly")
        assert len(anomalies) == 1
        payload = anomalies[0]["payload"]
        assert payload["detector"] == "exploding_grad"
        assert payload["signal"] == "grad_spike"
        assert payload["step"] == 7
    finally:
        await runner.stop()
        store.close()
