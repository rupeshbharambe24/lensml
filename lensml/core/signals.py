"""SIGUSR1 handler for pause / checkpoint / clean exit in trainer subprocess.

Plan refs:
    §3.1 — server signals trainer; trainer checkpoints model + optimizer +
           scheduler + RNG + sampler then exits cleanly.
    §8 Day 13 — checkpoint + restart implementation.
"""

from __future__ import annotations

# Implemented Day 13.
