"""Detector #8 — vanishing gradients (deep sigmoid networks).

Signal: per-layer grad norm — deepest < 1e-7 while shallowest is normal.
Fix:    `replace_activation.ReplaceActivation` (Sigmoid → GELU) plus a
        LayerNorm insertion.
Demo:   ⭐⭐⭐⭐ bar chart drama. (Plan §4 row 8.)
"""

from __future__ import annotations

# Implemented Day 18.
