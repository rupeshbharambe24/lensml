"""LowerLearningRate — fix for LR too high (oscillating loss).

Replaces `lr=L` with `lr=L*0.3` in the optimizer constructor call site.

Plan §4 row 2. Implemented Day 11.
"""

from __future__ import annotations
