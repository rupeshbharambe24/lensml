"""Detector #7 — bad initialization (plan §4 row 7).

Signal: at step 0 (or the first observed metric event), the
classification cross-entropy is more than ``multiplier`` × the
expected ``log(num_classes)``. A randomly-initialized K-class
classifier emits roughly uniform logits → CE ≈ log(K). A network
whose first-layer weights are too large will produce confidently
wrong predictions and a much higher CE.

The trainer opts in by publishing ``event["num_classes"]`` and
``event["loss"]`` on (or near) step 0. We fire the *first* time we
see a metric event with both fields and ``loss > multiplier *
log(num_classes)``.

Fix:    ``kaiming_init.KaimingInit`` — Kaiming-normal init for Linear
        and Conv2d layers via ``model.apply(...)``.
Demo:   ⭐⭐⭐ Karpathy classic.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

from lensml.detectors.base import Anomaly, Detector

DEFAULT_MULTIPLIER = 2.0
DEFAULT_MAX_STEP = 5  # only check the very first few steps


class BadInitDetector(Detector):
    name: ClassVar[str] = "bad_init"

    def __init__(
        self,
        *,
        multiplier: float = DEFAULT_MULTIPLIER,
        max_step: int = DEFAULT_MAX_STEP,
    ) -> None:
        self.multiplier = multiplier
        self.max_step = max_step
        self._fired = False
        self._past_window = False

    def observe(self, event: dict[str, Any]) -> Anomaly | None:
        if self._fired or self._past_window:
            return None
        if event.get("kind") != "metric":
            return None
        step = event.get("step")
        if not isinstance(step, int):
            return None
        if step > self.max_step:
            self._past_window = True
            return None

        loss = event.get("loss")
        num_classes = event.get("num_classes")
        if not isinstance(loss, (int, float)) or not isinstance(num_classes, int):
            return None
        loss_f = float(loss)
        if math.isnan(loss_f) or math.isinf(loss_f) or num_classes < 2:
            return None

        expected = math.log(num_classes)
        threshold = self.multiplier * expected
        if loss_f <= threshold:
            return None

        run_id = str(event.get("run_id", "default"))
        self._fired = True
        return Anomaly(
            detector=self.name,
            run_id=run_id,
            step=int(step),
            signal="loss_at_init_high",
            severity="warning",
            evidence={
                "loss": loss_f,
                "num_classes": int(num_classes),
                "expected_log_k": expected,
                "threshold": threshold,
                "ratio": loss_f / expected if expected > 0 else None,
            },
        )
