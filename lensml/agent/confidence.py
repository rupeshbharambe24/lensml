"""Confidence calibration via self-consistency.

Plan §5.4:
    N=3 samples at temperature 0.7. ≥2/3 agree → MEDIUM. 3/3 → HIGH.
    <2/3 → LOW; present as "I noticed X but I'm not sure" — no fix proposal.

Hard rule: never apply a fix below MEDIUM confidence without explicit user
override.
"""

from __future__ import annotations

# Implemented Day 10.
