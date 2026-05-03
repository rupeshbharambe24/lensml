"""lensml.replay — deterministic session playback (plan §3.6, §8 D20).

The replay subsystem powers the "bulletproof live-demo path": when the
network is hostile or the LLM is having an off day, we still need to be
able to walk a judge through the magic moment. A recorded session
streams through the same broker / SSE / dashboard pipeline as a live
trainer, so the demo looks identical to the real thing.

Public API:
    - ``Player`` — replays an iterable of events through a dispatch
      callable, pacing on the original ``ts`` deltas (or a fixed
      step delay).
    - ``read_jsonl`` — load a JSONL file into a list of event dicts.
    - ``write_jsonl`` — append an event dict as a JSONL line.
    - ``Recorder`` — context-manager wrapper that subscribes to a
      live ``EventBroker`` and writes every event to a JSONL file.
    - ``scenarios`` (submodule) — three deterministic demo scenarios
      (``exploding_grad``, ``dead_relu``, ``missing_normalize``).
"""

from __future__ import annotations

from lensml.replay.player import (
    DEFAULT_STEP_DELAY,
    Player,
    PlayerStats,
    read_jsonl,
    write_jsonl,
)
from lensml.replay.recorder import Recorder

__all__ = [
    "DEFAULT_STEP_DELAY",
    "Player",
    "PlayerStats",
    "Recorder",
    "read_jsonl",
    "write_jsonl",
]
