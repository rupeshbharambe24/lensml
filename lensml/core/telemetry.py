"""Metric extraction: loss, grad norms, activation mean/std/dead-fraction,
input batch stats, dataloader throughput, optimizer state.

Plan refs:
    §3 — telemetry feeds the agent's 12 read-only inspection tools (§5.2).
    §4 — detection signals depend on these metrics (e.g., grad-L2 rolling mean).
"""

from __future__ import annotations

# Implemented Day 2.
