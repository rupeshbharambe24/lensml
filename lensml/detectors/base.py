"""Base abstractions for stream detectors.

Plan refs:
    §4 — pathology catalogue. v1 ships 8 detectors; this module defines
         their common interface.
    §3.1 — server aggregates metrics; detectors are server-side stream
         consumers that emit `kind: "anomaly"` events back through the
         broker for the dashboard + agent.
    §5 — agent loop reacts to anomaly events; detector → agent is one
         direction of the data flow.

Detector interface contract:
    - Stateful: maintains rolling buffers / streak counters across calls.
    - Pure-functional w.r.t. side effects: `observe(event)` either returns
      an `Anomaly` or None — never publishes, never persists. The
      `DetectorRunner` is responsible for routing.
    - Robust: `observe()` must not raise for malformed events. Missing or
      non-numeric fields → return None.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar


@dataclass
class Anomaly:
    """Structured detection result.

    `signal` is a short slug identifying the trigger condition (e.g.
    ``"grad_spike"``, ``"nan_loss"``); the agent uses it to short-circuit
    diagnosis when the detector's evidence is already conclusive.
    `evidence` is a free-form dict of the values that fired the detector
    — used by the dashboard's AgentReasoning pane and by the LLM agent.
    """

    detector: str
    run_id: str
    step: int
    signal: str
    evidence: dict[str, Any]
    severity: str = "warning"
    ts: float = field(default_factory=time.time)

    def to_event(self) -> dict[str, Any]:
        """Serialize to the event-bus payload shape (`kind: "anomaly"`)."""
        return {
            "kind": "anomaly",
            "run_id": self.run_id,
            "step": self.step,
            "ts": self.ts,
            "detector": self.detector,
            "signal": self.signal,
            "severity": self.severity,
            "evidence": self.evidence,
        }


class Detector(ABC):
    """Base class for stream-processing detectors.

    Subclasses set `name` (used as the detector slug in anomaly events
    and in routing) and implement `observe()`.
    """

    name: ClassVar[str] = "base"

    @abstractmethod
    def observe(self, event: dict[str, Any]) -> Anomaly | None:
        """Process an incoming event. Return an Anomaly to fire, else None."""
        raise NotImplementedError
