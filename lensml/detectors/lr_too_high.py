"""Detector #2 — learning rate too high (oscillating loss).

Signal: sign-flip rate of Δloss > 0.6 over 50 steps.
Fix:    `lower_lr.LowerLearningRate` (multiply current LR by 0.3).
Demo:   ⭐⭐⭐⭐⭐ before/after curve. (Plan §4 row 2.)
"""

from __future__ import annotations

# Implemented Day 17.
