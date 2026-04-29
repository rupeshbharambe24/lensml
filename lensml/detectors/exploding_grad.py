"""Detector #1 — exploding gradients → NaN.

Signal: global grad-L2 > 10× rolling mean for ≥3 steps OR `loss.isnan()`.
Fix:    `grad_clip.AddGradientClipping` (max_norm=1.0).
Demo:   ⭐⭐⭐⭐⭐ visceral, <10s. (Plan §4 row 1.)
"""

from __future__ import annotations

# Implemented Day 6.
