"""Persistent + working memory for the agent loop.

Plan §5.3 — three layers:

    1. Working memory: last N=20 (action, observation) pairs verbatim.
    2. Sliding observations: older observations replaced with
       ``[masked: tool_name@step_X]``. Agent re-calls if it needs them.
    3. Persistent state: a structured ``DiagnosisState`` carried across
       cycles inside the system prompt.

No vector DB, no embeddings — bounded session, simple wins.

Informed by JetBrains' Dec 2025 context-rot findings: large unstructured
chat histories degrade tool-use accuracy on long traces. We keep the
prompt mostly structured (DiagnosisState in system) and only the recent
tool exchanges verbatim.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

DEFAULT_VERBATIM_PAIRS = 20


@dataclass
class ToolCall:
    """An action the agent took (a tool the LLM asked the runtime to run)."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Observation:
    """The runtime's response to a tool call (or an error)."""

    tool_name: str
    cycle: int  # which agent cycle this happened in
    result: dict[str, Any] | None = None
    error: str | None = None

    @property
    def is_error(self) -> bool:
        return self.error is not None

    def to_content(self) -> str:
        """Serialize for the OpenAI ``tool`` message ``content`` field."""
        if self.is_error:
            return json.dumps({"error": self.error})
        return json.dumps(self.result, default=str)


@dataclass
class DiagnosisState:
    """Structured agent state carried across the diagnose-anomaly cycles.

    Always rendered into the system prompt so the LLM sees its own
    progress without us replaying every tool exchange.
    """

    run_id: str
    anomaly: dict[str, Any] | None = None
    current_hypothesis: str | None = None
    confirmed: list[str] = field(default_factory=list)
    ruled_out: list[str] = field(default_factory=list)
    proposed_fixes: list[dict[str, Any]] = field(default_factory=list)
    confidence: str = "low"  # one of "low" | "medium" | "high"
    cycles_used: int = 0

    def summary_for_prompt(self) -> str:
        """Compact human-readable snapshot for the system prompt prefix."""
        lines: list[str] = []
        if self.current_hypothesis:
            lines.append(f"current hypothesis: {self.current_hypothesis}")
        if self.confirmed:
            lines.append("confirmed: " + ", ".join(self.confirmed))
        if self.ruled_out:
            lines.append("ruled out: " + ", ".join(self.ruled_out))
        if self.proposed_fixes:
            templates = [f.get("template", "?") for f in self.proposed_fixes]
            lines.append("proposed fixes: " + ", ".join(templates))
        if not lines:
            return "(no hypotheses yet)"
        return "\n".join(lines)


class History:
    """Bounded (action, observation) log with a sliding mask.

    Recent pairs render verbatim into the agent's message list; older
    pairs are replaced with a short masked placeholder so the agent
    knows a tool was called but won't burn context on its result.
    """

    __slots__ = ("_pairs", "max_verbatim")

    def __init__(self, max_verbatim: int = DEFAULT_VERBATIM_PAIRS) -> None:
        self.max_verbatim = max_verbatim
        self._pairs: list[tuple[ToolCall, Observation]] = []

    def __len__(self) -> int:
        return len(self._pairs)

    def append(self, call: ToolCall, obs: Observation) -> None:
        self._pairs.append((call, obs))

    @property
    def pairs(self) -> list[tuple[ToolCall, Observation]]:
        return list(self._pairs)

    def to_messages(self) -> list[dict[str, Any]]:
        """Render as OpenAI-compatible chat messages.

        For each pair we emit one assistant message (with ``tool_calls``)
        and one tool message (with the result/error). Older pairs beyond
        ``max_verbatim`` get masked content but keep the structural shape
        so the LLM still parses the tool-call protocol correctly.
        """
        if not self._pairs:
            return []
        cutoff = max(0, len(self._pairs) - self.max_verbatim)
        out: list[dict[str, Any]] = []
        for idx, (call, obs) in enumerate(self._pairs):
            masked = idx < cutoff
            content = (
                json.dumps({"masked": True, "tool": call.name, "cycle": obs.cycle})
                if masked
                else obs.to_content()
            )
            out.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments, default=str),
                            },
                        }
                    ],
                }
            )
            out.append({"role": "tool", "tool_call_id": call.id, "content": content})
        return out
