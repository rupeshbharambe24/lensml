"""Agent memory — three layers (plan §5.3).

    1. Working memory      last 20 (action, observation) pairs verbatim.
    2. Sliding observations older obs replaced with `[masked: tool@step_X]`;
                           agent re-calls if needed.
    3. Persistent state    structured `DiagnosisState`:
                           {current_hypotheses, confirmed, ruled_out,
                            proposed_fixes}. Lives in system prompt prefix.

No vector DB. Bounded session. Informed by JetBrains' Dec 2025 context-rot
findings.
"""

from __future__ import annotations

# Implemented Day 10.
