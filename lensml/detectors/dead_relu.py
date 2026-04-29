"""Detector #3 — dead ReLU on small MLP.

Signal: forward-hook dead fraction > 0.5 sustained.
Fix:    `replace_activation.ReplaceActivation` (ReLU → LeakyReLU(0.01)).
Demo:   ⭐⭐⭐⭐ animated dead-fraction meter. (Plan §4 row 3.)
"""

from __future__ import annotations

# Implemented Day 17.
