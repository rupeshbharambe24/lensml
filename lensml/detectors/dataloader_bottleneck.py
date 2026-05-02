"""Detector #5 — DataLoader bottleneck (plan §4 row 5).

Signal: trainer reports a high ratio of *batch wait time* to *step
time* for sustained steps, or equivalently a low GPU utilization
combined with ``num_workers == 0``. The trainer opts in by publishing
``event["dataloader_stats"] = {"batch_wait_ms": float,
"step_time_ms": float, "num_workers": int}``.

We fire single-shot when ``batch_wait_ms / step_time_ms`` is above
``ratio_threshold`` for ``persist_steps`` consecutive metric events,
OR when ``num_workers == 0`` and the wait ratio is above a softer
threshold.

Fix:    ``dataloader_workers.DataLoaderWorkers`` (set ``num_workers``,
        ``pin_memory``, ``persistent_workers``).
Demo:   ⭐⭐⭐ throughput jumps after fix.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

from lensml.detectors.base import Anomaly, Detector

DEFAULT_RATIO_THRESHOLD = 0.3
DEFAULT_NUM_WORKERS_RATIO_THRESHOLD = 0.15
DEFAULT_PERSIST_STEPS = 5


class DataloaderBottleneckDetector(Detector):
    name: ClassVar[str] = "dataloader_bottleneck"

    def __init__(
        self,
        *,
        ratio_threshold: float = DEFAULT_RATIO_THRESHOLD,
        num_workers_ratio_threshold: float = DEFAULT_NUM_WORKERS_RATIO_THRESHOLD,
        persist_steps: int = DEFAULT_PERSIST_STEPS,
    ) -> None:
        self.ratio_threshold = ratio_threshold
        self.num_workers_ratio_threshold = num_workers_ratio_threshold
        self.persist_steps = persist_steps
        self.streak = 0
        self._fired = False

    def observe(self, event: dict[str, Any]) -> Anomaly | None:
        if self._fired:
            return None
        if event.get("kind") != "metric":
            return None
        stats = event.get("dataloader_stats")
        if not isinstance(stats, dict):
            return None

        wait = stats.get("batch_wait_ms")
        step_t = stats.get("step_time_ms")
        if not isinstance(wait, (int, float)) or not isinstance(step_t, (int, float)):
            return None
        wait_f = float(wait)
        step_f = float(step_t)
        if step_f <= 0 or math.isnan(step_f) or math.isnan(wait_f) or math.isinf(step_f):
            return None
        ratio = wait_f / step_f

        nw = stats.get("num_workers")
        nw_int: int | None = int(nw) if isinstance(nw, int) else None

        threshold = self.num_workers_ratio_threshold if nw_int == 0 else self.ratio_threshold
        if ratio < threshold:
            self.streak = 0
            return None

        self.streak += 1
        if self.streak < self.persist_steps:
            return None

        run_id = str(event.get("run_id", "default"))
        step = int(event.get("step", 0)) if isinstance(event.get("step"), int) else 0
        self._fired = True
        return Anomaly(
            detector=self.name,
            run_id=run_id,
            step=step,
            signal="batch_wait_high",
            severity="warning",
            evidence={
                "wait_ratio": ratio,
                "threshold": threshold,
                "num_workers": nw_int,
                "batch_wait_ms": wait_f,
                "step_time_ms": step_f,
                "persist_steps": self.streak,
            },
        )
