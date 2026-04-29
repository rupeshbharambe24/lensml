"""Server-Sent Events helpers.

Provides ``EventBroker``, an in-memory async pub/sub used by the FastAPI
app to fan out incoming events (from the ZMQ bridge) to all connected
SSE clients.

Per-subscriber bounded queues prevent a slow client from causing memory
pressure on the server: when a queue fills, new events for that
subscriber are dropped (the loss is recoverable — clients can replay
from the persistent SQLite event log via the REST endpoint).

Plan refs:
    §3.1 — SSE via sse-starlette is the right transport for one-way
           server→browser streams. Native EventSource auto-reconnects.
    §3.2 — `server/sse.py` houses the SSE endpoints + helpers.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any


class EventBroker:
    """Async in-memory pub/sub for SSE fan-out.

    Producers call ``publish(event)``; each subscriber receives a copy via
    its own bounded ``asyncio.Queue``. ``put_nowait`` is used so a slow
    subscriber never blocks the producer — events for that subscriber
    are dropped silently when its queue is full.
    """

    __slots__ = ("_subscribers",)

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()

    def subscribe(self, maxsize: int = 256) -> asyncio.Queue[dict[str, Any]]:
        """Register a new subscriber queue. Caller is responsible for
        unsubscribing in a ``finally`` block to avoid leaks."""
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=maxsize)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        self._subscribers.discard(q)

    def publish(self, event: dict[str, Any]) -> int:
        """Fan out the event to all subscribers.

        Returns the number of successfully-delivered copies (drops counted
        in the difference: ``len(self._subscribers) - return_value``).
        """
        delivered = 0
        for q in self._subscribers:
            try:
                q.put_nowait(event)
                delivered += 1
            except asyncio.QueueFull:
                pass
        return delivered

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)


def format_sse(event: dict[str, Any]) -> str:
    """Serialize an event to the ``data: ...\\n\\n`` SSE line format.

    sse-starlette's ``EventSourceResponse`` does this for us when
    yielding dicts; this helper is for any low-level streaming uses
    (tests, custom endpoints).
    """
    return f"data: {json.dumps(event, default=str)}\n\n"
