"""Unit tests for lensml.server.sse."""

from __future__ import annotations

import asyncio
import json

import pytest

from lensml.server.sse import EventBroker, format_sse

# ---- format_sse ----


def test_format_sse_emits_data_line() -> None:
    out = format_sse({"step": 1, "loss": 0.5})
    assert out.startswith("data: ")
    assert out.endswith("\n\n")
    payload = json.loads(out[len("data: ") : -2])
    assert payload == {"step": 1, "loss": 0.5}


def test_format_sse_handles_non_serializable() -> None:
    class _C:
        def __str__(self) -> str:
            return "c"

    out = format_sse({"obj": _C()})
    assert "c" in out


# ---- EventBroker ----


@pytest.mark.asyncio
async def test_broker_starts_with_no_subscribers() -> None:
    b = EventBroker()
    assert b.subscriber_count == 0


@pytest.mark.asyncio
async def test_broker_subscribe_returns_queue() -> None:
    b = EventBroker()
    q = b.subscribe()
    assert isinstance(q, asyncio.Queue)
    assert b.subscriber_count == 1


@pytest.mark.asyncio
async def test_broker_unsubscribe_removes_queue() -> None:
    b = EventBroker()
    q = b.subscribe()
    b.unsubscribe(q)
    assert b.subscriber_count == 0


@pytest.mark.asyncio
async def test_broker_unsubscribe_idempotent() -> None:
    b = EventBroker()
    q = b.subscribe()
    b.unsubscribe(q)
    b.unsubscribe(q)


@pytest.mark.asyncio
async def test_broker_publishes_to_all_subscribers() -> None:
    b = EventBroker()
    q1 = b.subscribe()
    q2 = b.subscribe()
    delivered = b.publish({"x": 1})
    assert delivered == 2
    assert (await q1.get()) == {"x": 1}
    assert (await q2.get()) == {"x": 1}


@pytest.mark.asyncio
async def test_broker_publish_with_no_subscribers() -> None:
    b = EventBroker()
    delivered = b.publish({"x": 1})
    assert delivered == 0


@pytest.mark.asyncio
async def test_broker_drops_when_full() -> None:
    """A full subscriber queue drops events without blocking the producer."""
    b = EventBroker()
    q = b.subscribe(maxsize=2)
    assert b.publish({"i": 0}) == 1
    assert b.publish({"i": 1}) == 1
    # Third event finds queue full → drops, returns delivered=0
    assert b.publish({"i": 2}) == 0
    assert q.qsize() == 2


@pytest.mark.asyncio
async def test_broker_recovers_after_drain() -> None:
    """After consumer drains, publisher can deliver again."""
    b = EventBroker()
    q = b.subscribe(maxsize=2)
    b.publish({"i": 0})
    b.publish({"i": 1})
    b.publish({"i": 2})  # dropped
    await q.get()
    await q.get()
    assert b.publish({"i": 3}) == 1
