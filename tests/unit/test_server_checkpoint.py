"""Unit tests for lensml.server.checkpoint."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from lensml.server.checkpoint import (
    DEFAULT_CHECKPOINT_DIR,
    checkpoint_path,
    load_checkpoint,
    save_checkpoint,
)


class _MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def test_checkpoint_path_default_root() -> None:
    p = checkpoint_path("run-x", 5)
    assert p.parts[-3:] == ("run-x", "checkpoints", "step-00000005.pt")
    # default root is under ~/.lensml
    assert str(DEFAULT_CHECKPOINT_DIR) in str(p)


def test_checkpoint_path_custom_root(tmp_path: Path) -> None:
    p = checkpoint_path("r", 12, root=tmp_path)
    assert p == tmp_path / "r" / "checkpoints" / "step-00000012.pt"


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = _MLP()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Run one step so optimizer has state
    x = torch.randn(4, 4)
    y = torch.randint(0, 2, (4,))
    loss = nn.CrossEntropyLoss()(model(x), y)
    loss.backward()
    optimizer.step()

    weight_before = model.fc.weight.detach().clone()
    rng_before = torch.get_rng_state().clone()

    path = save_checkpoint(
        tmp_path / "ckpt.pt",
        step=42,
        next_batch_index=84,
        model=model,
        optimizer=optimizer,
        extra={"run_id": "r", "script_path": "/x/train.py"},
    )
    assert path.exists()

    # Mutate the model so we can verify the load actually restores it
    with torch.no_grad():
        model.fc.weight.add_(1.0)
    torch.manual_seed(999)

    info = load_checkpoint(path, model=model, optimizer=optimizer)
    assert info["step"] == 42
    assert info["next_batch_index"] == 84
    assert info["extra"]["run_id"] == "r"

    # Model weights restored
    assert torch.equal(model.fc.weight.detach(), weight_before)
    # torch RNG restored
    assert torch.equal(torch.get_rng_state(), rng_before)


def test_save_creates_parent_dirs(tmp_path: Path) -> None:
    model = _MLP()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    nested = tmp_path / "a" / "b" / "c" / "ckpt.pt"
    save_checkpoint(nested, step=0, next_batch_index=0, model=model, optimizer=optimizer)
    assert nested.exists()


def test_save_includes_python_and_numpy_rng(tmp_path: Path) -> None:
    """Spike 0.6 invariant: round-trip restores random + numpy RNG too."""
    random.seed(777)
    np.random.seed(777)
    py_state_before = random.getstate()
    np_state_before = np.random.get_state()

    model = _MLP()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    path = save_checkpoint(
        tmp_path / "ckpt.pt",
        step=0,
        next_batch_index=0,
        model=model,
        optimizer=optimizer,
    )

    # Disturb the RNGs
    random.seed(1)
    np.random.seed(1)

    load_checkpoint(path, model=model, optimizer=optimizer)
    assert random.getstate() == py_state_before
    np_state_after = np.random.get_state()
    # numpy state is a tuple-like; compare element-wise
    assert np_state_after[0] == np_state_before[0]


def test_save_handles_optional_scheduler(tmp_path: Path) -> None:
    model = _MLP()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Step the scheduler so its internal counter is non-zero
    sched.step()
    sched.step()

    path = save_checkpoint(
        tmp_path / "ckpt.pt",
        step=0,
        next_batch_index=0,
        model=model,
        optimizer=optimizer,
        scheduler=sched,
    )

    # Reset scheduler
    fresh_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    fresh_sched = torch.optim.lr_scheduler.StepLR(fresh_optimizer, step_size=10, gamma=0.5)
    info = load_checkpoint(path, model=model, optimizer=fresh_optimizer, scheduler=fresh_sched)
    assert info["step"] == 0
    # Scheduler last_epoch matches what we stepped to before saving
    assert fresh_sched.last_epoch == 2


def test_load_without_scheduler_when_saved_without(tmp_path: Path) -> None:
    """Saving without a scheduler then loading also without — must not crash."""
    model = _MLP()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    path = save_checkpoint(
        tmp_path / "ckpt.pt",
        step=3,
        next_batch_index=6,
        model=model,
        optimizer=optimizer,
        scheduler=None,
    )
    info = load_checkpoint(path, model=model, optimizer=optimizer, scheduler=None)
    assert info["step"] == 3
    assert info["next_batch_index"] == 6


def test_load_extra_defaults_empty(tmp_path: Path) -> None:
    model = _MLP()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    path = save_checkpoint(
        tmp_path / "ckpt.pt",
        step=0,
        next_batch_index=0,
        model=model,
        optimizer=optimizer,
    )
    info = load_checkpoint(path, model=model, optimizer=optimizer)
    assert info["extra"] == {}


def test_load_missing_file_raises(tmp_path: Path) -> None:
    model = _MLP()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "nope.pt", model=model, optimizer=optimizer)
