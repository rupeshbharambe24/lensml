"""Integration tests: lensml LensmlCallback (step 3.1).

Mirrors the raw-PyTorch coverage from step 2.7 but binds to the
Lightning callback lifecycle:
    - on_train_start sets up run_dir / PID file / signal
    - on_before_optimizer_step publishes telemetry + honors stop signal
    - on_save_checkpoint injects lensml metadata
    - on_load_checkpoint restores step + next_batch_index
    - on_train_end tears down hooks + PID file + publisher
    - lensml_resume_ckpt() reads LENSML_RESUME

Tests are skipped if pytorch-lightning isn't installed (it's an
optional extra).
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

pl = pytest.importorskip("lightning.pytorch")

from lensml.core.signals import EXIT_CHECKPOINTED  # noqa: E402
from lensml.core.transport import JSONLPublisher  # noqa: E402
from lensml.integrations.lightning import (  # noqa: E402
    LENSML_CKPT_KEY,
    RESUME_ENV_VAR,
    LensmlCallback,
    lensml_resume_ckpt,
)


class _LitMLP(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)
        self.relu = nn.ReLU()  # so install_capture_hooks has something to bind to

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.relu(x))

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        return nn.functional.cross_entropy(self(x), y)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.01)


def _make_publisher() -> JSONLPublisher:
    return JSONLPublisher(stream=io.StringIO())


def _make_loader(n: int = 16) -> DataLoader:
    x = torch.randn(n, 4)
    y = torch.randint(0, 2, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=4)


def _make_trainer(tmp_path: Path, callbacks: list, max_steps: int = 4) -> pl.Trainer:
    return pl.Trainer(
        max_steps=max_steps,
        accelerator="cpu",
        devices=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        enable_checkpointing=False,
        default_root_dir=str(tmp_path / "lit"),
        callbacks=callbacks,
    )


# ---- helper ----


def test_lensml_resume_ckpt_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(RESUME_ENV_VAR, raising=False)
    assert lensml_resume_ckpt() is None


def test_lensml_resume_ckpt_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(RESUME_ENV_VAR, "/tmp/foo.ckpt")
    assert lensml_resume_ckpt() == "/tmp/foo.ckpt"


def test_lensml_resume_ckpt_empty_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(RESUME_ENV_VAR, "")
    assert lensml_resume_ckpt() is None


# ---- run_dir / PID file lifecycle ----


def test_callback_writes_pid_file(tmp_path: Path) -> None:
    cb = LensmlCallback(
        run_id="r",
        run_dir=tmp_path / "run",
        publisher=_make_publisher(),
    )
    trainer = _make_trainer(tmp_path, callbacks=[cb])
    trainer.fit(_LitMLP(), _make_loader())
    # PID file is removed on train_end, but the run_dir + checkpoints subdir
    # must still exist after fit.
    assert (tmp_path / "run" / "checkpoints").is_dir()
    assert not (tmp_path / "run" / "trainer.pid").exists()


def test_callback_pid_file_exists_during_training(tmp_path: Path) -> None:
    """Spy on on_train_start to verify the PID file is written before the
    first batch and only removed on train_end."""
    seen: dict[str, bool] = {}

    class _Spy(pl.Callback):
        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            pid_file = tmp_path / "run" / "trainer.pid"
            seen["mid"] = pid_file.exists() and pid_file.read_text(encoding="utf-8") == str(
                os.getpid()
            )

    cb = LensmlCallback(run_id="r", run_dir=tmp_path / "run", publisher=_make_publisher())
    trainer = _make_trainer(tmp_path, callbacks=[cb, _Spy()], max_steps=2)
    trainer.fit(_LitMLP(), _make_loader())

    assert seen.get("mid") is True
    assert not (tmp_path / "run" / "trainer.pid").exists()


def test_enable_checkpoint_false_skips_filesystem(tmp_path: Path) -> None:
    cb = LensmlCallback(
        run_id="r",
        run_dir=tmp_path / "run",
        enable_checkpoint=False,
        publisher=_make_publisher(),
    )
    trainer = _make_trainer(tmp_path, callbacks=[cb], max_steps=2)
    trainer.fit(_LitMLP(), _make_loader())

    assert not (tmp_path / "run" / "trainer.pid").exists()
    assert cb._signal is None


# ---- step counter ----


def test_step_counter_advances_per_batch(tmp_path: Path) -> None:
    cb = LensmlCallback(run_id="r", run_dir=tmp_path / "run", publisher=_make_publisher())
    trainer = _make_trainer(tmp_path, callbacks=[cb], max_steps=3)
    trainer.fit(_LitMLP(), _make_loader())
    assert cb.step == 3
    assert cb.next_batch_index == 3


# ---- on_save / on_load checkpoint metadata ----


def test_save_checkpoint_injects_lensml_metadata(tmp_path: Path) -> None:
    cb = LensmlCallback(run_id="r-ckpt", run_dir=tmp_path / "run", publisher=_make_publisher())
    trainer = _make_trainer(tmp_path, callbacks=[cb], max_steps=2)
    trainer.fit(_LitMLP(), _make_loader())

    # Manually save a checkpoint via the callback API to inspect contents.
    ckpt_path = tmp_path / "saved.ckpt"
    trainer.save_checkpoint(str(ckpt_path))
    blob = torch.load(str(ckpt_path), weights_only=False)
    assert LENSML_CKPT_KEY in blob
    meta = blob[LENSML_CKPT_KEY]
    assert meta["run_id"] == "r-ckpt"
    assert meta["step"] == 2
    assert meta["next_batch_index"] == 2


def test_load_checkpoint_restores_counters(tmp_path: Path) -> None:
    """Round-trip: save a Lightning ckpt with our callback, then resume
    a fresh callback via Trainer.fit(ckpt_path=...) and verify the
    counters are restored + resumed=True."""
    cb1 = LensmlCallback(run_id="r-orig", run_dir=tmp_path / "run1", publisher=_make_publisher())
    trainer1 = _make_trainer(tmp_path, callbacks=[cb1], max_steps=3)
    model = _LitMLP()
    trainer1.fit(model, _make_loader())
    assert cb1.step == 3

    ckpt_path = tmp_path / "round_trip.ckpt"
    trainer1.save_checkpoint(str(ckpt_path))

    # Build a fresh callback + trainer and resume.
    cb2 = LensmlCallback(
        run_id="placeholder", run_dir=tmp_path / "run2", publisher=_make_publisher()
    )
    # Lightning needs strictly more max_steps than the resumed step.
    trainer2 = _make_trainer(tmp_path, callbacks=[cb2], max_steps=4)
    trainer2.fit(_LitMLP(), _make_loader(), ckpt_path=str(ckpt_path))

    assert cb2.resumed is True
    assert cb2.run_id == "r-orig"
    # After resume + 1 more step, step counter should be 4.
    assert cb2.step == 4


# ---- stop signal triggers checkpoint + sys.exit(99) ----


def test_stop_signal_triggers_checkpoint_and_exit(tmp_path: Path) -> None:
    """Pre-trigger the signal and then run training; the first
    on_before_optimizer_step must save a Lightning checkpoint and
    sys.exit(EXIT_CHECKPOINTED)."""
    cb = LensmlCallback(run_id="r", run_dir=tmp_path / "run", publisher=_make_publisher())
    trainer = _make_trainer(tmp_path, callbacks=[cb], max_steps=10)

    class _Trigger(pl.Callback):
        def on_train_start(self, trainer, pl_module):
            assert cb._signal is not None
            cb._signal.trigger()

    trainer = _make_trainer(tmp_path, callbacks=[cb, _Trigger()], max_steps=10)

    with pytest.raises(SystemExit) as exc:
        trainer.fit(_LitMLP(), _make_loader())
    assert exc.value.code == EXIT_CHECKPOINTED

    ckpt_dir = tmp_path / "run" / "checkpoints"
    assert ckpt_dir.is_dir()
    assert any(p.suffix == ".ckpt" for p in ckpt_dir.iterdir())


def test_stop_flag_file_triggers_checkpoint_and_exit(tmp_path: Path) -> None:
    cb = LensmlCallback(run_id="r", run_dir=tmp_path / "run", publisher=_make_publisher())

    class _Trigger(pl.Callback):
        def on_train_start(self, trainer, pl_module):
            (tmp_path / "run" / "STOP").touch()

    trainer = _make_trainer(tmp_path, callbacks=[cb, _Trigger()], max_steps=10)
    with pytest.raises(SystemExit) as exc:
        trainer.fit(_LitMLP(), _make_loader())
    assert exc.value.code == EXIT_CHECKPOINTED


# ---- manual checkpoint via cb.checkpoint() ----


def test_manual_checkpoint_writes_file(tmp_path: Path) -> None:
    cb = LensmlCallback(run_id="r", run_dir=tmp_path / "run", publisher=_make_publisher())
    trainer = _make_trainer(tmp_path, callbacks=[cb], max_steps=2)
    trainer.fit(_LitMLP(), _make_loader())

    path = cb.checkpoint(trainer)
    assert path.exists()
    assert path.suffix == ".ckpt"
    assert path.parent.name == "checkpoints"
