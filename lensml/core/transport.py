"""ZMQ PUB/SUB transport between trainer subprocess and FastAPI server.

Plan refs:
    §3.1 — pyzmq PUB/SUB is ~5 lines, survives either side restarting,
           microsecond latency. The trainer is the publisher; the server's
           ZMQ-SUB side is wired to SSE for the browser dashboard.
    §3.1 — Inproc fallback for testing: stdout JSONL when ZMQ isn't
           available or when piping into another tool.

Endpoint conventions:
    - Production default: ``tcp://127.0.0.1:5555``. Cross-platform.
    - Tests / in-process: ``inproc://lensml-test``. No port allocation,
      no slow-joiner sleep needed (inproc still has the issue but it's
      tighter; tests use a brief poll loop).
    - Plan §3.1 mentions ``ipc://`` — pyzmq ipc support on Windows is
      flaky; we default to TCP for portability.

Slow-joiner caveat (PUB/SUB):
    Messages sent before the subscriber connects ARE LOST. Production
    flow: start the lensml server first (subscriber-side), then start
    the trainer (publisher-side). Document in user-facing docs.
"""

from __future__ import annotations

import json
import os
import sys
from typing import IO, Any

import zmq

DEFAULT_ENDPOINT = "tcp://127.0.0.1:5555"
DEFAULT_TIMEOUT_MS = 1000


class JSONLPublisher:
    """Fallback transport: line-delimited JSON to a writable stream.

    Used when ZMQ isn't available or for piping into another tool. Default
    stream is ``sys.stdout``. Implements the same minimal interface as
    ``ZMQPublisher``: ``publish(event_dict)`` and ``close()``.
    """

    __slots__ = ("_owns_stream", "stream")

    def __init__(self, stream: IO[str] | None = None) -> None:
        self.stream = stream if stream is not None else sys.stdout
        self._owns_stream = False

    def publish(self, event: dict[str, Any]) -> None:
        self.stream.write(json.dumps(event, default=str) + "\n")
        self.stream.flush()

    def close(self) -> None:
        # Don't close caller-provided streams (especially stdout).
        if self._owns_stream:
            self.stream.close()


class ZMQPublisher:
    """Trainer-side ZMQ PUB socket.

    By default binds to the given endpoint (the trainer is the producer
    and exposes a published port). Set ``bind=False`` to connect instead
    (rare; typically when the server is the bind side and you want the
    trainer to connect into a known address).
    """

    __slots__ = ("_owns_ctx", "ctx", "endpoint", "sock")

    def __init__(
        self,
        endpoint: str = DEFAULT_ENDPOINT,
        *,
        bind: bool = True,
        ctx: zmq.Context | None = None,
    ) -> None:
        self._owns_ctx = ctx is None
        self.ctx = ctx if ctx is not None else zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.PUB)
        if bind:
            self.sock.bind(endpoint)
        else:
            self.sock.connect(endpoint)
        last = self.sock.last_endpoint
        self.endpoint = last.decode() if isinstance(last, bytes) else last or endpoint

    def publish(self, event: dict[str, Any]) -> None:
        self.sock.send_json(event)

    def close(self) -> None:
        self.sock.close(linger=0)
        # Don't terminate a context we didn't create — others may share it.


class ZMQSubscriber:
    """Server-side / test-side ZMQ SUB socket.

    Connects to the given endpoint and subscribes to all topics by default.
    """

    __slots__ = ("_owns_ctx", "ctx", "sock")

    def __init__(
        self,
        endpoint: str = DEFAULT_ENDPOINT,
        *,
        topics: tuple[bytes, ...] = (b"",),
        ctx: zmq.Context | None = None,
    ) -> None:
        self._owns_ctx = ctx is None
        self.ctx = ctx if ctx is not None else zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.connect(endpoint)
        for topic in topics:
            self.sock.setsockopt(zmq.SUBSCRIBE, topic)

    def recv(self, timeout_ms: int = DEFAULT_TIMEOUT_MS) -> dict[str, Any] | None:
        """Poll for one message; return ``None`` on timeout."""
        if self.sock.poll(timeout_ms):
            return self.sock.recv_json()
        return None

    def close(self) -> None:
        self.sock.close(linger=0)


def make_publisher(endpoint: str | None = None) -> JSONLPublisher | ZMQPublisher:
    """Factory: pick a publisher based on ``endpoint`` or ``LENSML_TRANSPORT``.

    Resolution order:
        1. Explicit ``endpoint`` argument.
        2. ``LENSML_TRANSPORT`` env var.
        3. Default: ``DEFAULT_ENDPOINT`` (ZMQ TCP localhost).

    Special values:
        - ``"jsonl"`` → ``JSONLPublisher`` writing to stdout.
        - Any zmq-style URL (``tcp://``, ``ipc://``, ``inproc://``)
          → ``ZMQPublisher``.
    """
    if endpoint is None:
        endpoint = os.environ.get("LENSML_TRANSPORT", DEFAULT_ENDPOINT)
    if endpoint == "jsonl":
        return JSONLPublisher()
    return ZMQPublisher(endpoint)
