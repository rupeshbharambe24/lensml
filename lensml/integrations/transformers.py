"""HuggingFace Trainer integration — `LensmlTrainerCallback`.

Plan refs:
    §2 — HF TrainerCallback is read-only (cannot mutate the loop).
         `TrainerControl.should_training_stop = True` is the supported escape
         hatch. lensml's pattern: callback signals stop → external orchestrator
         applies fix → restart Trainer with `resume_from_checkpoint`.
    §8 Day 16 — implementation; test on DistilBERT/SST-2 (Colab T4).
    §11 risk #1 — high probability that subprocess restart with full state
         continuity is harder than expected; fallback ships HF as
         "callback mode only, restart is manual" and documents the gap.
"""

from __future__ import annotations


class LensmlTrainerCallback:
    """HuggingFace Trainer callback. Use with `Trainer(callbacks=[...])`."""

    # Implemented Day 16 — see lensml-plan.md §8.
