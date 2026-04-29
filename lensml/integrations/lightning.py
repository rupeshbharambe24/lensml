"""PyTorch Lightning integration — `LensmlCallback`.

Plan refs:
    §2 — Lightning callbacks expose on_train_batch_start/end,
         on_before_optimizer_step, on_after_backward, on_save_checkpoint,
         on_load_checkpoint. Trainer.fit(ckpt_path=...) resumes with full state.
    §8 Day 15 — implementation; test on CIFAR-10 + small CNN.
"""

from __future__ import annotations


class LensmlCallback:
    """Lightning callback. Pass to `pl.Trainer(callbacks=[LensmlCallback()])`."""

    # Implemented Day 15 — see lensml-plan.md §8.
