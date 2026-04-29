"""lensml — an open-source AI agent that supervises ML training runs in real time.

Public API (per plan §3.2 / §5):
    lensml.watch(model, optimizer, dataloader, **kwargs)  -> WatchHandle
    lensml.unwatch(handle)                                -> None

Convenience re-exports:
    lensml.LensmlCallback         (PyTorch Lightning integration)
    lensml.LensmlTrainerCallback  (HuggingFace Trainer integration)
"""

from __future__ import annotations

__version__ = "0.1.0"

# NOTE: The implementations are stubbed during scaffolding (see day-by-day
# build calendar in lensml-plan.md §8). Real bodies land Days 5+.
from lensml.integrations.pytorch import unwatch, watch

__all__ = [
    "__version__",
    "unwatch",
    "watch",
]
