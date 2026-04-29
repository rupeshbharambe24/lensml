"""Unit tests for lensml.core.transport."""

from __future__ import annotations

import io
import json

import pytest
import zmq

from lensml.core.transport import (
    DEFAULT_ENDPOINT,
    JSONLPublisher,
    ZMQPublisher,
    ZMQSubscriber,
    make_publisher,
)

# ---- JSONLPublisher ----


def test_jsonl_publisher_writes_to_stream() -> None:
    stream = io.StringIO()
    pub = JSONLPublisher(stream=stream)
    pub.publish({"step": 1, "loss": 0.5})
    pub.publish({"step": 2, "loss": 0.4})
    pub.close()

    lines = stream.getvalue().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"step": 1, "loss": 0.5}
    assert json.loads(lines[1]) == {"step": 2, "loss": 0.4}


def test_jsonl_publisher_flushes_each_event() -> None:
    """Each publish() should flush so consumers see events immediately."""
    stream = io.StringIO()
    pub = JSONLPublisher(stream=stream)
    pub.publish({"x": 1})
    # If we didn't flush, getvalue() might still see the line because StringIO
    # has no buffering, but we want the contract to hold for any IO subclass.
    assert "x" in stream.getvalue()


def test_jsonl_publisher_handles_non_serializable_via_str() -> None:
    """Non-JSON-serializable values fall back to str() via default=str."""
    stream = io.StringIO()
    pub = JSONLPublisher(stream=stream)

    class _Custom:
        def __str__(self) -> str:
            return "custom-repr"

    pub.publish({"obj": _Custom()})
    parsed = json.loads(stream.getvalue().strip())
    assert parsed["obj"] == "custom-repr"


def test_jsonl_publisher_does_not_close_caller_stream() -> None:
    """When the caller passes their own stream, close() must not close it."""
    stream = io.StringIO()
    pub = JSONLPublisher(stream=stream)
    pub.close()
    assert not stream.closed


# ---- ZMQ pub/sub round trip via inproc ----


@pytest.fixture
def zmq_ctx():
    ctx = zmq.Context()
    yield ctx
    ctx.term()


def test_zmq_inproc_round_trip(zmq_ctx: zmq.Context) -> None:
    endpoint = "inproc://lensml-test-roundtrip"
    pub = ZMQPublisher(endpoint, ctx=zmq_ctx)
    sub = ZMQSubscriber(endpoint, ctx=zmq_ctx)

    # Slow-joiner: PUB/SUB inproc still has the same issue. Send a few times
    # and let the subscriber poll until something arrives.
    received = None
    for _ in range(20):
        pub.publish({"step": 7, "loss": 1.5})
        msg = sub.recv(timeout_ms=50)
        if msg is not None:
            received = msg
            break

    assert received == {"step": 7, "loss": 1.5}

    sub.close()
    pub.close()


def test_zmq_subscriber_returns_none_on_timeout(zmq_ctx: zmq.Context) -> None:
    endpoint = "inproc://lensml-test-timeout"
    pub = ZMQPublisher(endpoint, ctx=zmq_ctx)
    sub = ZMQSubscriber(endpoint, ctx=zmq_ctx)

    # No messages ever published — subscriber should time out cleanly.
    assert sub.recv(timeout_ms=50) is None

    sub.close()
    pub.close()


def test_zmq_publisher_endpoint_attribute(zmq_ctx: zmq.Context) -> None:
    pub = ZMQPublisher("inproc://lensml-test-endpoint", ctx=zmq_ctx)
    assert pub.endpoint == "inproc://lensml-test-endpoint"
    pub.close()


# ---- make_publisher factory ----


def test_make_publisher_jsonl_keyword() -> None:
    pub = make_publisher("jsonl")
    assert isinstance(pub, JSONLPublisher)
    pub.close()


def test_make_publisher_zmq_url(zmq_ctx: zmq.Context) -> None:
    pub = make_publisher("inproc://lensml-test-factory")
    assert isinstance(pub, ZMQPublisher)
    pub.close()


def test_make_publisher_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LENSML_TRANSPORT", "jsonl")
    pub = make_publisher()
    assert isinstance(pub, JSONLPublisher)
    pub.close()


def test_make_publisher_default_endpoint() -> None:
    """Default constant matches the documented value."""
    assert DEFAULT_ENDPOINT == "tcp://127.0.0.1:5555"
