"""Detector #8 — vanishing gradients (plan §4 row 8).

Signal: per-layer gradient L2 norms — the deepest layer (last entry
in insertion order) has grad < ``floor`` while the shallowest layer's
grad is significantly larger. Sustained over ``persist_steps``.
Pattern of deep sigmoid / tanh networks where chain-ruled gradients
shrink exponentially with depth.

The trainer opts in by publishing ``event["per_layer_grad_l2"]``
(an ordered dict from input-side to output-side, or vice versa — the
detector treats min/max symmetrically). lensml's ``per_layer_grad_l2``
helper in ``lensml/core/telemetry.py`` produces exactly this shape.

Fire when ``min(grad_l2_values) < floor`` AND ``max/min ratio``
exceeds ``ratio_threshold`` for ``persist_steps`` consecutive events.

Fix:    ``replace_activation.ReplaceActivation`` (Sigmoid → GELU)
        plus optional LayerNorm.
Demo:   ⭐⭐⭐⭐ bar chart drama.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

from lensml.detectors.base import Anomaly, Detector

DEFAULT_FLOOR = 1e-7
DEFAULT_RATIO_THRESHOLD = 100.0
DEFAULT_PERSIST_STEPS = 3


class VanishingGradDetector(Detector):
    name: ClassVar[str] = "vanishing_grad"

    def __init__(
        self,
        *,
        floor: float = DEFAULT_FLOOR,
        ratio_threshold: float = DEFAULT_RATIO_THRESHOLD,
        persist_steps: int = DEFAULT_PERSIST_STEPS,
    ) -> None:
        self.floor = floor
        self.ratio_threshold = ratio_threshold
        self.persist_steps = persist_steps
        self.streak = 0
        self._last_min_layer: str | None = None
        self._last_max_layer: str | None = None
        self._last_min_value: float = 0.0
        self._last_max_value: float = 0.0

    def observe(self, event: dict[str, Any]) -> Anomaly | None:
        if event.get("kind") != "metric":
            return None
        per_layer = event.get("per_layer_grad_l2")
        if not isinstance(per_layer, dict) or not per_layer:
            return None

        # Pull the numeric grad-L2 values, keyed by layer name.
        valid: list[tuple[str, float]] = []
        for k, v in per_layer.items():
            if isinstance(v, (int, float)):
                vf = float(v)
                if not math.isnan(vf) and not math.isinf(vf):
                    valid.append((str(k), vf))
        if len(valid) < 2:
            return None

        # Identify the smallest and largest layer grads.
        min_layer, min_val = min(valid, key=lambda kv: kv[1])
        max_layer, max_val = max(valid, key=lambda kv: kv[1])

        if min_val >= self.floor:
            self.streak = 0
            return None
        if max_val <= 0:
            return None
        ratio = max_val / min_val if min_val > 0 else float("inf")
        if ratio < self.ratio_threshold:
            self.streak = 0
            return None

        self.streak += 1
        self._last_min_layer = min_layer
        self._last_max_layer = max_layer
        self._last_min_value = min_val
        self._last_max_value = max_val
        if self.streak < self.persist_steps:
            return None

        run_id = str(event.get("run_id", "default"))
        step = int(event.get("step", 0)) if isinstance(event.get("step"), int) else 0
        anomaly = Anomaly(
            detector=self.name,
            run_id=run_id,
            step=step,
            signal="grad_vanishing",
            severity="warning",
            evidence={
                "min_layer": min_layer,
                "min_grad_l2": min_val,
                "max_layer": max_layer,
                "max_grad_l2": max_val,
                "ratio": ratio,
                "floor": self.floor,
                "persist_steps": self.streak,
            },
        )
        self.streak = 0
        return anomaly
