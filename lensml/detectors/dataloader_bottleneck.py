"""Detector #5 — DataLoader bottleneck.

Signal: GPU idle fraction > 30% sustained AND `num_workers=0` confirmed.
Fix:    `dataloader_workers.DataLoaderWorkers` — sets num_workers,
        pin_memory=True, persistent_workers=True.
Demo:   ⭐⭐⭐ throughput 3×. (Plan §4 row 5.)
"""

from __future__ import annotations

# Implemented Day 18.
