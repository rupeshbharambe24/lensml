"""AddLRScheduler — fix for missing learning-rate schedule.

Inserts a simple `torch.optim.lr_scheduler.CosineAnnealingLR` (or similar) and
the corresponding `scheduler.step()` call.

Plan §3.2 (template list). Listed for v1 even though §4's pathology table
focuses on `lower_lr`; this template is a follow-on for the LR-schedule
mismatch class flagged in §6.1's stretch list.
"""

from __future__ import annotations
