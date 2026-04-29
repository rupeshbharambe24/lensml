"""ReplaceActivation — fix for dead ReLU and vanishing-grad sigmoid.

ReLU → LeakyReLU(0.01)         (dead ReLU, plan §4 row 3)
Sigmoid → GELU + LayerNorm     (vanishing grads, plan §4 row 8)

Implemented Day 11.
"""

from __future__ import annotations
