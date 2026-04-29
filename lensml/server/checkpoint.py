"""Checkpoint save/load — model + optimizer + scheduler + RNG + sampler.

Plan §3.3 — directory layout `~/.lensml/runs/{run_id}/checkpoints/{step}/`:
    model.pt        state_dict
    optimizer.pt    optimizer state
    scheduler.pt    if present
    rng.pt          CPU + CUDA + numpy + python random
    sampler.pt      DataLoader sampler state — best effort
    meta.json       epoch, global_step, wall_time

Plan §11 risk #1: subprocess restart with full state continuity (RNG,
dataloader sampler) has known edge cases. Day 13 spike; fallback is to skip
restart and tell the user to restart manually.
"""

from __future__ import annotations

# Implemented Day 13.
