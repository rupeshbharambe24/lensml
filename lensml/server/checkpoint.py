"""Checkpoint save / load — plan §3.3 schema validated by spike 0.6.

Spike 0.6 confirmed (notes/restart-spike-2026-04-29.md): the contents
listed below are sufficient to give byte-identical loss trajectories
across kill + restart on both CPU and GPU.

Stored payload:
    step                   — int, the global optimizer step number
    next_batch_index       — int, position in the (deterministic) dataset
    model                  — nn.Module.state_dict()
    optimizer              — torch.optim.Optimizer.state_dict()
    scheduler              — LRScheduler.state_dict() | None
    torch_rng_state        — torch.get_rng_state()
    cuda_rng_state         — torch.cuda.get_rng_state_all() (CUDA only)
    python_rng_state       — random.getstate()
    numpy_rng_state        — np.random.get_state()
    meta                   — {wall_time, lensml_version}
    extra                  — caller-supplied dict (run_id, script_path, etc.)
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    import torch.nn as nn

DEFAULT_CHECKPOINT_DIR = Path.home() / ".lensml" / "runs"


def checkpoint_path(run_id: str, step: int, *, root: Path | None = None) -> Path:
    """Canonical path for a checkpoint of `run_id` at `step`."""
    base = root if root is not None else DEFAULT_CHECKPOINT_DIR
    return base / run_id / "checkpoints" / f"step-{step:08d}.pt"


def save_checkpoint(
    path: Path,
    *,
    step: int,
    next_batch_index: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Persist trainer state to `path`. Creates parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "step": int(step),
        "next_batch_index": int(next_batch_index),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "torch_rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "meta": {
            "wall_time": time.time(),
        },
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    if extra:
        payload["extra"] = dict(extra)

    torch.save(payload, path)
    return path


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None = None,
) -> dict[str, Any]:
    """Restore from `path`. Returns ``{"step", "next_batch_index", "extra"}``."""
    payload = torch.load(path, weights_only=False)

    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])

    torch.set_rng_state(payload["torch_rng_state"])
    if "cuda_rng_state" in payload and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
    random.setstate(payload["python_rng_state"])
    np.random.set_state(payload["numpy_rng_state"])

    return {
        "step": int(payload["step"]),
        "next_batch_index": int(payload["next_batch_index"]),
        "extra": dict(payload.get("extra", {})),
    }
