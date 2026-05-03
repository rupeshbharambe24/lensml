"""Recorder — write every event from a live broker to a JSONL file.

Plan §3.6: the replay system needs *something* to replay. A recorder
sits next to the dashboard during a real training run and persists the
event stream to disk. Later, ``Player`` reads that file back.

Subscribes to an ``EventBroker`` via the standard async queue API,
reads forever, and appends each event to ``path`` until ``stop()`` is
called or the surrounding async context is cancelled.

Use as an async context manager::

    async with Recorder(broker, path) as rec:
        ...           # run the trainer
    # rec.events_written = total
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lensml.replay.player import write_jsonl

if TYPE_CHECKING:
    from lensml.server.sse import EventBroker

log = logging.getLogger(__name__)


class Recorder:
    """Persists an EventBroker's stream to a JSONL file."""

    def __init__(
        self,
        broker: EventBroker,
        path: Path,
        *,
        run_id: str | None = None,
    ) -> None:
        self.broker = broker
        self.path = path
        self.run_id = run_id
        self.events_written = 0
        self._task: asyncio.Task[None] | None = None
        self._queue: asyncio.Queue[dict[str, Any]] | None = None

    async def start(self) -> None:
        if self._task is not None:
            return
        # Truncate the target so re-using a path produces a fresh recording.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")
        self._queue = self.broker.subscribe()
        self._task = asyncio.create_task(self._run(), name="lensml-recorder")

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

    async def __aenter__(self) -> Recorder:
        await self.start()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.stop()

    async def _run(self) -> None:
        assert self._queue is not None
        try:
            while True:
                event = await self._queue.get()
                if self.run_id is not None and event.get("run_id") not in (None, self.run_id):
                    # Filter to just the run we're recording.
                    continue
                try:
                    write_jsonl(self.path, event)
                    self.events_written += 1
                except OSError:
                    log.exception("recorder: write to %s failed", self.path)
        except asyncio.CancelledError:
            pass
