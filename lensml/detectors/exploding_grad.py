"""Detector #1 — exploding gradients → NaN.

Plan §4 row 1:
    Signal: global grad-L2 > 10× rolling mean for ≥3 steps OR `loss.isnan()`.
    Fix:    `grad_clip.AddGradientClipping` (max_norm=1.0).
    Demo:   ⭐⭐⭐⭐⭐ visceral, <10 s.

Implementation details:
    - Rolling history holds non-spike values only — including spike values
      would inflate the threshold and mask sustained problems.
    - `streak` counts consecutive spike steps; resets to 0 on a non-spike
      step OR after firing (so a sustained anomaly only re-fires after
      another `persist_steps` spikes).
    - NaN loss is a separate, faster trigger: fires immediately, no
      warmup needed.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, ClassVar

from lensml.detectors.base import Anomaly, Detector

DEFAULT_WINDOW = 30
DEFAULT_WARMUP = 10
DEFAULT_MULTIPLIER = 10.0
DEFAULT_PERSIST_STEPS = 3


class ExplodingGradDetector(Detector):
    name: ClassVar[str] = "exploding_grad"

    def __init__(
        self,
        *,
        window: int = DEFAULT_WINDOW,
        warmup: int = DEFAULT_WARMUP,
        multiplier: float = DEFAULT_MULTIPLIER,
        persist_steps: int = DEFAULT_PERSIST_STEPS,
    ) -> None:
        self.window = window
        self.warmup = warmup
        self.multiplier = multiplier
        self.persist_steps = persist_steps
        self.history: deque[float] = deque(maxlen=window)
        self.streak = 0

    def observe(self, event: dict[str, Any]) -> Anomaly | None:
        if event.get("kind") != "metric":
            return None

        run_id = str(event.get("run_id", "default"))
        step = int(event.get("step", 0)) if isinstance(event.get("step"), int) else 0

        # Fast path: NaN loss is conclusive on its own — no warmup needed.
        loss = event.get("loss")
        if isinstance(loss, float) and math.isnan(loss):
            return Anomaly(
                detector=self.name,
                run_id=run_id,
                step=step,
                signal="nan_loss",
                severity="critical",
                evidence={"loss": "NaN"},
            )

        grad_l2 = event.get("grad_l2")
        if not isinstance(grad_l2, (int, float)):
            return None
        grad_l2 = float(grad_l2)
        if grad_l2 < 0 or math.isnan(grad_l2) or math.isinf(grad_l2):
            # Inf grad-L2 is also exploding — fire immediately.
            if math.isinf(grad_l2):
                return Anomaly(
                    detector=self.name,
                    run_id=run_id,
                    step=step,
                    signal="inf_grad",
                    severity="critical",
                    evidence={"grad_l2": "Inf"},
                )
            return None

        # Warmup: collect baseline before we can compute a meaningful threshold.
        if len(self.history) < self.warmup:
            self.history.append(grad_l2)
            return None

        rolling_mean = sum(self.history) / len(self.history)
        # Guard against degenerate baseline (all zeros) — multiplier * 0 == 0,
        # which would flag every nonzero grad as a spike. Require an absolute
        # floor so the detector behaves sensibly on under-fitting models.
        threshold = max(rolling_mean * self.multiplier, 1e-6)

        if grad_l2 > threshold:
            self.streak += 1
            if self.streak >= self.persist_steps:
                anomaly = Anomaly(
                    detector=self.name,
                    run_id=run_id,
                    step=step,
                    signal="grad_spike",
                    severity="critical",
                    evidence={
                        "grad_l2": grad_l2,
                        "rolling_mean": rolling_mean,
                        "ratio": grad_l2 / rolling_mean if rolling_mean > 0 else float("inf"),
                        "persist_steps": self.streak,
                        "threshold": threshold,
                    },
                )
                self.streak = 0  # require another `persist_steps` to re-fire
                return anomaly
            return None

        # Non-spike value: reset streak, update baseline.
        self.streak = 0
        self.history.append(grad_l2)
        return None
