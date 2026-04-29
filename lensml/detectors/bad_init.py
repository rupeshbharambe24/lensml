"""Detector #7 — bad initialization (loss-at-init wrong).

Signal: at step 0, K-class cross-entropy expected ≈ log(K); observed > 2×.
Fix:    `kaiming_init.KaimingInit` — `model.apply(_init)` with Kaiming for
        ReLU layers.
Demo:   ⭐⭐⭐ Karpathy classic. (Plan §4 row 7.)
"""

from __future__ import annotations

# Implemented Day 18.
