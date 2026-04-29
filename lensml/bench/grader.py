"""Deterministic per-fixture grader.

Plan §6.1 — each fixture has a `grade.py` with three checks:
    (a) does patched code run without error?
    (b) does loss converge below threshold T within N steps?
    (c) does test metric exceed threshold M?

Plan §6.4 — top-1 / top-3 diagnosis accuracy, pass@1 / pass@3 fix rate,
time-to-diagnose, tokens-to-diagnose, cost-to-diagnose.
"""

from __future__ import annotations

# Implemented Day 23.
