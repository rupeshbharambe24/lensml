"""Detector #3 — dead ReLU (plan §4 row 3).

Signal: any captured activation has ``dead_fraction > threshold`` for
``persist_steps`` consecutive ``kind: "metric"`` events that include
activation stats. Reads ``event["activations"][name]["dead_fraction"]``
which is what ``activation_stats()`` (plan §3.1) emits.

Fix:    ``replace_activation.ReplaceActivation`` (ReLU → LeakyReLU(0.01)).
Demo:   ⭐⭐⭐⭐ animated dead-fraction meter.

Like ``ExplodingGradDetector``, we use a streak counter to require
persistence — a single bad batch isn't enough. Tracks the *worst*
module per step so the firing evidence points at the offender.
"""

from __future__ import annotations

from typing import Any, ClassVar

from lensml.detectors.base import Anomaly, Detector

DEFAULT_THRESHOLD = 0.5
DEFAULT_PERSIST_STEPS = 5


class DeadReluDetector(Detector):
    name: ClassVar[str] = "dead_relu"

    def __init__(
        self,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        persist_steps: int = DEFAULT_PERSIST_STEPS,
    ) -> None:
        self.threshold = threshold
        self.persist_steps = persist_steps
        self.streak = 0
        self._worst_name: str | None = None
        self._worst_value: float = 0.0

    def observe(self, event: dict[str, Any]) -> Anomaly | None:
        if event.get("kind") != "metric":
            return None
        activations = event.get("activations")
        if not isinstance(activations, dict) or not activations:
            return None

        # Find the most-dead module in this event.
        worst_name: str | None = None
        worst_dead: float = -1.0
        for mod_name, stats in activations.items():
            if not isinstance(stats, dict):
                continue
            dead = stats.get("dead_fraction")
            if not isinstance(dead, (int, float)):
                continue
            dead_f = float(dead)
            if dead_f > worst_dead:
                worst_dead = dead_f
                worst_name = str(mod_name)

        if worst_name is None or worst_dead < 0:
            return None

        if worst_dead > self.threshold:
            self.streak += 1
            self._worst_name = worst_name
            self._worst_value = worst_dead
            if self.streak >= self.persist_steps:
                run_id = str(event.get("run_id", "default"))
                step = int(event.get("step", 0)) if isinstance(event.get("step"), int) else 0
                anomaly = Anomaly(
                    detector=self.name,
                    run_id=run_id,
                    step=step,
                    signal="dead_fraction_high",
                    severity="warning",
                    evidence={
                        "module": worst_name,
                        "dead_fraction": worst_dead,
                        "threshold": self.threshold,
                        "persist_steps": self.streak,
                    },
                )
                self.streak = 0
                return anomaly
            return None

        # Below threshold — reset streak.
        self.streak = 0
        return None
