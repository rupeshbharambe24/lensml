"""Benchmark runner — orchestrates fixture execution.

Plan §6 / §8 Day 24:
    Spawns subprocesses per fixture, runs grader, aggregates results.
    Idempotent: rerunning skips fixtures with passing results unless --force.

Plan §11 risk #6: Colab/Kaggle session disconnects during bench runs (high P,
low impact). Mitigated by per-fixture state checkpointing and re-running
only failed fixtures.
"""

from __future__ import annotations

# Implemented Day 24.
