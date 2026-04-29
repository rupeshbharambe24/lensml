"""ReAct loop — ~80 LOC target.

Plan refs:
    §5.1 — pseudocode for the agent loop. Cap MAX_CYCLES at 8 per anomaly.
    §8 Day 10 — highest-risk day of the agent build. Fallback: skip
                self-consistency for v1, ship single-shot.
"""

from __future__ import annotations

# Implemented Day 10.
