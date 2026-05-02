"""Integration tests: lensml.watch() + checkpoint + signal plumbing (step 2.7).

Covers the wiring added in step 2.7:
    - LENSML_RESUME env var loads a checkpoint into the new handle
    - PID file is written on _install, removed on unwatch
    - The CheckpointSignal is reachable through handle._signal and triggers
      sys.exit(EXIT_CHECKPOINTED) at the next optimizer.step
    - handle.checkpoint() saves a valid file the next handle can resume from
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from lensml.core.signals import EXIT_CHECKPOINTED
from lensml.core.transport import JSONLPublisher
from lensml.integrations.pytorch import RESUME_ENV_VAR, watch


class _MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _make_publisher() -> JSONLPublisher:
    return JSONLPublisher(stream=io.StringIO())


def _train_step(model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    optimizer.zero_grad(set_to_none=True)
    x = torch.randn(4, 4)
    y = torch.randint(0, 2, (4,))
    loss = nn.CrossEntropyLoss()(model(x), y)
    loss.backward()
    optimizer.step()


# ---- run_dir + PID file ----


def test_install_writes_pid_file(tmp_path: Path) -> None:
    model = _MLP()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    h = watch(model, opt, run_id="r", run_dir=tmp_path / "run", publisher=_make_publisher())

    pid_file = tmp_path / "run" / "trainer.pid"
    assert pid_file.exists()
    assert pid_file.read_text(encoding="utf-8") == str(os.getpid())

    h.unwatch()


def test_unwatch_removes_pid_file(tmp_path: Path) -> None:
    model = _MLP()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    h = watch(model, opt, run_id="r", run_dir=tmp_path / "run", publisher=_make_publisher())
    pid_file = tmp_path / "run" / "trainer.pid"
    assert pid_file.exists()
    h.unwatch()
    assert not pid_file.exists()


def test_install_creates_checkpoints_subdir(tmp_path: Path) -> None:
    model = _MLP()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    run_dir = tmp_path / "run"
    h = watch(model, opt, run_id="r", run_dir=run_dir, publisher=_make_publisher())
    assert (run_dir / "checkpoints").is_dir()
    h.unwatch()


def test_enable_checkpoint_false_skips_filesystem(tmp_path: Path) -> None:
    model = _MLP()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    h = watch(
        model,
        opt,
        run_id="r",
        run_dir=tmp_path / "run",
        enable_checkpoint=False,
        publisher=_make_publisher(),
    )
    # No PID file when checkpointing is disabled
    assert not (tmp_path / "run" / "trainer.pid").exists()
    assert h._signal is None
    h.unwatch()


# ---- handle.checkpoint() ----


def test_checkpoint_method_writes_file(tmp_path: Path) -> None:
    model = _MLP()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    h = watch(model, opt, run_id="r", run_dir=tmp_path / "run", publisher=_make_publisher())

    # Run a couple of steps so step counter advances
    for _ in range(3):
        _train_step(model, opt)
    assert h.step == 3

    path = h.checkpoint()
    assert path.exists()
    assert path.parent.name == "checkpoints"
    assert "step-00000003" in path.name

    h.unwatch()


# ---- LENSML_RESUME ----


def test_resume_loads_step_and_next_batch_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round-trip: handle.checkpoint() then a new watch() with LENSML_RESUME
    set must restore step + next_batch_index."""
    # First handle: train a few steps and checkpoint
    model = _MLP()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    h1 = watch(model, opt, run_id="r", run_dir=tmp_path / "run1", publisher=_make_publisher())
    for _ in range(5):
        _train_step(model, opt)
    assert h1.step == 5
    ckpt_path = h1.checkpoint()
    h1.unwatch()

    # Capture weights so we can verify they're restored on resume
    weight_after = model.fc.weight.detach().clone()

    # Mutate the model — load_checkpoint should overwrite this
    with torch.no_grad():
        model.fc.weight.add_(10.0)

    # New handle with LENSML_RESUME pointing at the saved checkpoint
    monkeypatch.setenv(RESUME_ENV_VAR, str(ckpt_path))
    fresh_opt = torch.optim.SGD(model.parameters(), lr=0.01)
    h2 = watch(
        model,
        fresh_opt,
        run_id="r-resumed",
        run_dir=tmp_path / "run2",
        publisher=_make_publisher(),
    )
    assert h2.resumed is True
    assert h2.step == 5
    assert h2.next_batch_index == 5
    # Weights restored
    assert torch.equal(model.fc.weight.detach(), weight_after)
    h2.unwatch()


def test_resume_with_missing_file_does_not_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If LENSML_RESUME points at a nonexistent file, watch() warns + continues
    fresh rather than crashing."""
    monkeypatch.setenv(RESUME_ENV_VAR, str(tmp_path / "missing.pt"))
    model = _MLP()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    h = watch(model, opt, run_id="r", run_dir=tmp_path / "run", publisher=_make_publisher())
    assert h.resumed is False
    assert h.step == 0
    h.unwatch()


def test_no_resume_when_env_unset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(RESUME_ENV_VAR, raising=False)
    model = _MLP()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    h = watch(model, opt, run_id="r", run_dir=tmp_path / "run", publisher=_make_publisher())
    assert h.resumed is False
    h.unwatch()


# ---- signal handler triggering exit ----


def test_stop_signal_triggers_sys_exit_99(tmp_path: Path) -> None:
    """Programmatic .trigger() on the signal must cause the next
    optimizer.step to checkpoint + sys.exit(EXIT_CHECKPOINTED)."""
    model = _MLP()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    h = watch(model, opt, run_id="r", run_dir=tmp_path / "run", publisher=_make_publisher())

    # Trigger the stop signal directly (bypasses needing a real SIGUSR1)
    assert h._signal is not None
    h._signal.trigger()

    with pytest.raises(SystemExit) as exc:
        _train_step(model, opt)
    assert exc.value.code == EXIT_CHECKPOINTED

    # A checkpoint should have been written before exit
    ckpt_dir = tmp_path / "run" / "checkpoints"
    assert any(p.name.startswith("step-") for p in ckpt_dir.iterdir())


def test_stop_flag_file_triggers_checkpoint_and_exit(tmp_path: Path) -> None:
    model = _MLP()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    run_dir = tmp_path / "run"
    watch(model, opt, run_id="r", run_dir=run_dir, publisher=_make_publisher())

    # The server side would touch this file; we do it directly here.
    (run_dir / "STOP").touch()

    with pytest.raises(SystemExit) as exc:
        _train_step(model, opt)
    assert exc.value.code == EXIT_CHECKPOINTED


def test_no_signal_when_checkpoint_disabled(tmp_path: Path) -> None:
    """With enable_checkpoint=False, no signal handler is set up, so a
    triggered signal would have no effect (and there's nothing to
    trigger)."""
    model = _MLP()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    h = watch(
        model,
        opt,
        run_id="r",
        run_dir=tmp_path / "run",
        enable_checkpoint=False,
        publisher=_make_publisher(),
    )
    # Should be able to run normal steps without any exit
    _train_step(model, opt)
    assert h.step == 1
    h.unwatch()
