"""Detector #2 — learning rate too high / oscillating loss (plan §4 row 2).

Signal: sign-flip rate of Δloss > ``flip_ratio`` over a rolling window
of ``window`` consecutive losses (default: ratio 0.6 over 50 steps).
Healthy training has small, infrequent sign flips; a too-high LR makes
loss bounce around the basin and produces a flip on most steps.

Fix:    ``lower_lr.LowerLearningRate`` (multiply current LR by 0.3).
Demo:   ⭐⭐⭐⭐⭐ before/after curve.

Reads ``event["loss"]`` (user-published via ``handle.log(loss=...)``
or via the integrations' callback metric publishing).
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, ClassVar

from lensml.detectors.base import Anomaly, Detector

DEFAULT_WINDOW = 50
DEFAULT_FLIP_RATIO = 0.6
DEFAULT_COOLDOWN = 50  # don't re-fire while still observing the same window


class LrTooHighDetector(Detector):
    name: ClassVar[str] = "lr_too_high"

    def __init__(
        self,
        *,
        window: int = DEFAULT_WINDOW,
        flip_ratio: float = DEFAULT_FLIP_RATIO,
        cooldown: int = DEFAULT_COOLDOWN,
    ) -> None:
        self.window = window
        self.flip_ratio = flip_ratio
        self.cooldown = cooldown
        self.losses: deque[float] = deque(maxlen=window)
        self._cooldown_remaining = 0

    def observe(self, event: dict[str, Any]) -> Anomaly | None:
        if event.get("kind") != "metric":
            return None
        loss = event.get("loss")
        if not isinstance(loss, (int, float)):
            return None
        loss_f = float(loss)
        if math.isnan(loss_f) or math.isinf(loss_f):
            return None

        self.losses.append(loss_f)

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return None

        if len(self.losses) < self.window:
            return None

        # Compute sign-flip rate of consecutive Δloss.
        deltas = [self.losses[i + 1] - self.losses[i] for i in range(len(self.losses) - 1)]
        flips = 0
        for i in range(1, len(deltas)):
            # Strict opposite signs only — a 0 delta breaks the chain rather
            # than counting as a flip.
            if (deltas[i - 1] > 0 and deltas[i] < 0) or (deltas[i - 1] < 0 and deltas[i] > 0):
                flips += 1
        # Possible flip count is len(deltas) - 1 = window - 2.
        possible = len(deltas) - 1
        if possible <= 0:
            return None
        ratio = flips / possible
        if ratio < self.flip_ratio:
            return None

        run_id = str(event.get("run_id", "default"))
        step = int(event.get("step", 0)) if isinstance(event.get("step"), int) else 0
        lr = event.get("lr") if isinstance(event.get("lr"), (int, float)) else None
        anomaly = Anomaly(
            detector=self.name,
            run_id=run_id,
            step=step,
            signal="loss_oscillation",
            severity="warning",
            evidence={
                "flip_ratio": ratio,
                "threshold": self.flip_ratio,
                "window": self.window,
                "lr": float(lr) if lr is not None else None,
            },
        )
        self._cooldown_remaining = self.cooldown
        return anomaly
