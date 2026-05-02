"""DetectorRunner — connects detectors to the server's event broker.

Plan §3.1:
    The server aggregates metrics. Detectors run server-side, subscribed to
    the broker; when one fires, the runner publishes the anomaly back
    through the broker (so the dashboard sees it via SSE) AND persists it
    to the SQLite event log (so history survives restarts).

Why a separate runner instead of running detectors in `_zmq_bridge`:
    - Single responsibility: bridge handles transport ingress; runner handles
      detection. Either can be tested in isolation.
    - Detectors can be added/removed at runtime without touching transport.
    - Future agent loop (plan §5) is another consumer of the same broker —
      same pattern.

The runner's own anomaly events come back to its own queue (since it's a
broker subscriber). It ignores them by filtering on `kind == "metric"` —
no infinite loop.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING

from lensml.detectors.base import Detector
from lensml.server.sse import EventBroker
from lensml.server.store import EventStore

if TYPE_CHECKING:
    from typing import Any

log = logging.getLogger(__name__)


class DetectorRunner:
    """Runs a set of detectors against the broker's event stream."""

    def __init__(
        self,
        broker: EventBroker,
        store: EventStore,
        detectors: list[Detector],
    ) -> None:
        self.broker = broker
        self.store = store
        self.detectors = detectors
        self._task: asyncio.Task[None] | None = None
        self._queue: asyncio.Queue[dict[str, Any]] | None = None

    async def start(self) -> None:
        if self._task is not None:
            return
        self._queue = self.broker.subscribe()
        self._task = asyncio.create_task(self._run(), name="lensml-detector-runner")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await self._task
        self._task = None
        if self._queue is not None:
            self.broker.unsubscribe(self._queue)
            self._queue = None

    async def _run(self) -> None:
        assert self._queue is not None
        try:
            while True:
                event = await self._queue.get()
                # Detectors only act on raw metric events; we ignore our own
                # anomaly emissions and the run_started/run_ended brackets.
                if event.get("kind") != "metric":
                    continue
                for detector in self.detectors:
                    try:
                        anomaly = detector.observe(event)
                    except Exception:
                        log.exception("detector %s.observe raised", detector.name)
                        continue
                    if anomaly is None:
                        continue
                    payload = anomaly.to_event()
                    try:
                        self.store.append_event(anomaly.run_id, "anomaly", payload)
                    except Exception:
                        log.exception("event store append failed for anomaly")
                    self.broker.publish(payload)
        except asyncio.CancelledError:
            pass
