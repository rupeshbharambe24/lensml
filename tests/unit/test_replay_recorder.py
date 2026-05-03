"""Unit tests for lensml.replay.recorder."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from lensml.replay import Recorder, read_jsonl
from lensml.server.sse import EventBroker


@pytest.mark.asyncio
async def test_recorder_writes_published_events(tmp_path: Path) -> None:
    broker = EventBroker()
    out = tmp_path / "rec.jsonl"
    async with Recorder(broker, out) as rec:
        broker.publish({"kind": "metric", "run_id": "r", "step": 0, "loss": 0.5})
        broker.publish({"kind": "anomaly", "run_id": "r", "step": 1})
        # Yield the loop so the recorder task drains the queue.
        await asyncio.sleep(0.05)
        assert rec.events_written == 2

    events = read_jsonl(out)
    assert [e["kind"] for e in events] == ["metric", "anomaly"]


@pytest.mark.asyncio
async def test_recorder_filters_by_run_id(tmp_path: Path) -> None:
    broker = EventBroker()
    out = tmp_path / "rec.jsonl"
    async with Recorder(broker, out, run_id="target") as rec:
        broker.publish({"kind": "metric", "run_id": "target", "step": 0})
        broker.publish({"kind": "metric", "run_id": "other", "step": 0})
        broker.publish({"kind": "metric", "run_id": "target", "step": 1})
        await asyncio.sleep(0.05)
        assert rec.events_written == 2

    events = read_jsonl(out)
    assert {e["run_id"] for e in events} == {"target"}


@pytest.mark.asyncio
async def test_recorder_truncates_existing_file(tmp_path: Path) -> None:
    out = tmp_path / "rec.jsonl"
    out.write_text('{"stale": true}\n', encoding="utf-8")
    broker = EventBroker()
    async with Recorder(broker, out):
        broker.publish({"kind": "metric", "step": 0})
        await asyncio.sleep(0.05)

    events = read_jsonl(out)
    assert len(events) == 1
    assert events[0]["kind"] == "metric"


@pytest.mark.asyncio
async def test_recorder_stop_is_idempotent(tmp_path: Path) -> None:
    broker = EventBroker()
    rec = Recorder(broker, tmp_path / "rec.jsonl")
    await rec.start()
    await rec.stop()
    await rec.stop()  # second stop must not raise


@pytest.mark.asyncio
async def test_recorder_unsubscribes_on_stop(tmp_path: Path) -> None:
    broker = EventBroker()
    rec = Recorder(broker, tmp_path / "rec.jsonl")
    await rec.start()
    assert broker.subscriber_count == 1
    await rec.stop()
    assert broker.subscriber_count == 0
