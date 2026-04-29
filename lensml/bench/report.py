"""Benchmark report generator.

Plan §6.5 — `bench/results/{date}/report.md` + `report.json`. Includes:
    - Per-class confusion matrix (where the agent confuses pathologies)
    - Latency histogram
    - Cost breakdown (always $0 — make this a slide)
    - "What doesn't work yet" section — honest about which classes lensml
      fails on. Japanese judges (Round 3) respect this.
"""

from __future__ import annotations

# Implemented Day 26.
