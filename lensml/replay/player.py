"""Replay player — paces a stream of events through a dispatch callable.

Plan §3.6, §8 D20. Two pacing modes:

    1. ``ts``-driven: if both consecutive events have a ``ts`` field, sleep
       ``ts[i] - ts[i-1]`` seconds (divided by ``speed``). This preserves
       the original cadence of a recorded session, so a 100-step run
       that took 30 s plays back in 30 s.

    2. fixed: when an event lacks ``ts`` (or ``ts`` runs backwards), fall
       back to ``fixed_delay`` seconds between events. Synthetic
       scenarios use this path.

The player never touches HTTP / sockets directly. The caller passes
``dispatch(event)`` and is responsible for routing the event somewhere
(an in-process broker, an HTTP POST, a stdout dump, …). This keeps the
core deterministic and unit-testable without a live server.

For ``run_id`` rewriting (so a replayed demo doesn't collide with a
real run in the dashboard), pass ``run_id`` to ``Player`` — every
event's ``run_id`` field is overwritten before dispatch.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_STEP_DELAY = 0.05  # 50 ms ≈ 20 Hz, matching plan §3.1's tick envelope
MAX_SLEEP_SECONDS = 5.0  # cap pause between events to keep demos snappy

DispatchFn = Callable[[dict[str, Any]], Awaitable[None]] | Callable[[dict[str, Any]], None]


@dataclass
class PlayerStats:
    """Tally of what was dispatched during a play() run."""

    dispatched: int = 0
    skipped: int = 0
    elapsed: float = 0.0
    kinds: dict[str, int] = field(default_factory=dict)

    def record(self, event: dict[str, Any]) -> None:
        self.dispatched += 1
        kind = str(event.get("kind", "unknown"))
        self.kinds[kind] = self.kinds.get(kind, 0) + 1


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file. Skips blank lines and lines that fail to parse."""
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                log.warning("replay: skipping malformed JSONL line in %s", path)
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


def write_jsonl(path: Path, event: dict[str, Any]) -> None:
    """Append a single event to ``path`` as a JSON line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, default=str))
        f.write("\n")


def _pacing_iter(
    events: Iterable[dict[str, Any]],
    *,
    fixed_delay: float,
    speed: float,
) -> Iterator[tuple[dict[str, Any], float]]:
    """Yield (event, sleep_before_dispatch_seconds) pairs.

    The sleep is the gap from the *previous* event to this one, scaled
    by ``speed`` and clamped to ``[0, MAX_SLEEP_SECONDS]``.
    """
    prev_ts: float | None = None
    first = True
    for event in events:
        ts = event.get("ts")
        ts_f = float(ts) if isinstance(ts, (int, float)) else None
        if first:
            sleep = 0.0
            first = False
        elif ts_f is not None and prev_ts is not None and ts_f > prev_ts:
            sleep = (ts_f - prev_ts) / max(speed, 1e-9)
        else:
            sleep = fixed_delay / max(speed, 1e-9)
        sleep = max(0.0, min(MAX_SLEEP_SECONDS, sleep))
        yield event, sleep
        if ts_f is not None:
            prev_ts = ts_f


class Player:
    """Replays a sequence of events through a user-supplied dispatch.

    Use ``play()`` for blocking sync replay (suitable for the CLI when
    the dispatch is sync HTTP or file I/O), or ``play_async()`` for
    asyncio-driven replay (when the dispatch awaits, e.g. an in-process
    ``EventBroker.publish`` + ``store.append_event``).
    """

    def __init__(
        self,
        *,
        fixed_delay: float = DEFAULT_STEP_DELAY,
        speed: float = 1.0,
        run_id: str | None = None,
    ) -> None:
        if speed <= 0:
            raise ValueError("speed must be > 0")
        if fixed_delay < 0:
            raise ValueError("fixed_delay must be >= 0")
        self.fixed_delay = fixed_delay
        self.speed = speed
        self.run_id = run_id

    def _prepare(self, event: dict[str, Any]) -> dict[str, Any]:
        if self.run_id is None:
            return event
        # Shallow copy + override; keeps the original event dict in caller's
        # iterable untouched.
        out = dict(event)
        out["run_id"] = self.run_id
        return out

    async def play_async(
        self,
        events: Iterable[dict[str, Any]],
        dispatch: DispatchFn,
    ) -> PlayerStats:
        stats = PlayerStats()
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        for event, sleep in _pacing_iter(events, fixed_delay=self.fixed_delay, speed=self.speed):
            if sleep > 0:
                await asyncio.sleep(sleep)
            prepared = self._prepare(event)
            try:
                result = dispatch(prepared)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                log.exception("replay: dispatch raised — continuing")
                stats.skipped += 1
                continue
            stats.record(prepared)
        stats.elapsed = loop.time() - t0
        return stats

    def play(
        self,
        events: Iterable[dict[str, Any]],
        dispatch: Callable[[dict[str, Any]], None],
    ) -> PlayerStats:
        """Synchronous replay. ``dispatch`` MUST be sync."""
        import time

        stats = PlayerStats()
        t0 = time.time()
        for event, sleep in _pacing_iter(events, fixed_delay=self.fixed_delay, speed=self.speed):
            if sleep > 0:
                time.sleep(sleep)
            prepared = self._prepare(event)
            try:
                dispatch(prepared)
            except Exception:
                log.exception("replay: dispatch raised — continuing")
                stats.skipped += 1
                continue
            stats.record(prepared)
        stats.elapsed = time.time() - t0
        return stats
