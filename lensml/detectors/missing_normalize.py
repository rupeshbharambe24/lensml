"""Detector #4 — missing input normalization (plan §4 row 4).

Signal: the first observed batch input mean is outside ``[-3, 3]`` (or
the first input has unusually large standard deviation), suggesting
the trainer is feeding raw uint8 / [0, 255] image data straight into
the network without subtracting a mean and dividing by a std.

Fix:    ``add_normalize.AddNormalize`` (ImageNet stats by default).
Demo:   ⭐⭐⭐⭐ accuracy jumps after fix.

Reads ``event["input_stats"]`` — a small opt-in payload the trainer
publishes once per training. Concretely:
    handle.log(input_stats={"mean": x.mean().item(),
                            "std": x.std().item(),
                            "min": x.min().item(),
                            "max": x.max().item()})

Single-shot: the detector fires at most once per run (cooldown after
fire so a re-running trainer doesn't get spammed).
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

from lensml.detectors.base import Anomaly, Detector

DEFAULT_MEAN_BOUND = 3.0
DEFAULT_MAX_BOUND = 50.0  # data-range upper bound for typical normalized images


class MissingNormalizeDetector(Detector):
    name: ClassVar[str] = "missing_normalize"

    def __init__(
        self,
        *,
        mean_bound: float = DEFAULT_MEAN_BOUND,
        max_bound: float = DEFAULT_MAX_BOUND,
    ) -> None:
        self.mean_bound = mean_bound
        self.max_bound = max_bound
        self._fired = False

    def observe(self, event: dict[str, Any]) -> Anomaly | None:
        if self._fired:
            return None
        if event.get("kind") != "metric":
            return None
        stats = event.get("input_stats")
        if not isinstance(stats, dict):
            return None

        mean = stats.get("mean")
        if not isinstance(mean, (int, float)):
            return None
        mean_f = float(mean)
        if math.isnan(mean_f) or math.isinf(mean_f):
            return None

        max_v = stats.get("max")
        max_f = float(max_v) if isinstance(max_v, (int, float)) else None
        min_v = stats.get("min")
        min_f = float(min_v) if isinstance(min_v, (int, float)) else None

        # Trigger conditions (any one is sufficient — they're all symptoms of
        # raw / unnormalized data):
        triggers: list[str] = []
        if abs(mean_f) > self.mean_bound:
            triggers.append("mean_out_of_range")
        if max_f is not None and max_f > self.max_bound:
            triggers.append("max_out_of_range")
        # Image data on [0, 1] post-normalize sits roughly in [-2.5, 2.5].
        # If both min ≥ 0 AND max > 1.5, it's almost certainly unnormalized.
        if min_f is not None and max_f is not None and min_f >= 0.0 and max_f > 1.5:
            triggers.append("unsigned_range")

        if not triggers:
            return None

        run_id = str(event.get("run_id", "default"))
        step = int(event.get("step", 0)) if isinstance(event.get("step"), int) else 0
        self._fired = True
        return Anomaly(
            detector=self.name,
            run_id=run_id,
            step=step,
            signal="unnormalized_input",
            severity="warning",
            evidence={
                "input_mean": mean_f,
                "input_max": max_f,
                "input_min": min_f,
                "triggers": triggers,
                "mean_bound": self.mean_bound,
                "max_bound": self.max_bound,
            },
        )
