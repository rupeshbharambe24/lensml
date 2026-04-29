"""Detector #4 — missing input normalization.

Signal: first-batch input mean ∉ [-3, 3] for image tensor.
Fix:    `add_normalize.AddNormalize` (ImageNet stats by default).
Demo:   ⭐⭐⭐⭐ accuracy jumps after fix. (Plan §4 row 4.)
"""

from __future__ import annotations

# Implemented Day 17.
