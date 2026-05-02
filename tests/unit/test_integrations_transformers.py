"""Integration tests: lensml LensmlTrainerCallback (step 3.2).

HF TrainerCallback is read-only — stop is requested via
``control.should_training_stop = True`` rather than by raising. Tests
exercise the callback both via direct hook invocation (the
TrainerControl plumbing) and via a small end-to-end fit on a tiny
model + a real ``Trainer`` to verify save-on-stop and the
``EXIT_CHECKPOINTED`` exit code.

Tests are skipped if transformers isn't installed (optional extra).
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

transformers = pytest.importorskip("transformers")

from transformers import (  # noqa: E402
    Trainer,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from lensml.core.signals import EXIT_CHECKPOINTED  # noqa: E402
from lensml.core.transport import JSONLPublisher  # noqa: E402
from lensml.integrations.transformers import (  # noqa: E402
    RESUME_ENV_VAR,
    LensmlTrainerCallback,
    lensml_resume_ckpt,
)


def _make_publisher() -> JSONLPublisher:
    return JSONLPublisher(stream=io.StringIO())


# ---- helper ----


def test_lensml_resume_ckpt_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(RESUME_ENV_VAR, raising=False)
    assert lensml_resume_ckpt() is None


def test_lensml_resume_ckpt_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(RESUME_ENV_VAR, "/tmp/checkpoint-100")
    assert lensml_resume_ckpt() == "/tmp/checkpoint-100"


def test_lensml_resume_ckpt_empty_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(RESUME_ENV_VAR, "")
    assert lensml_resume_ckpt() is None


# ---- direct-hook unit tests (no Trainer) ----


def _make_state(global_step: int = 0) -> TrainerState:
    s = TrainerState()
    s.global_step = global_step
    return s


def _make_args(tmp_path: Path) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(tmp_path / "out"),
        report_to="none",
    )


def test_on_train_begin_writes_pid_file(tmp_path: Path) -> None:
    cb = LensmlTrainerCallback(run_id="r", run_dir=tmp_path / "run", publisher=_make_publisher())
    args = _make_args(tmp_path)
    state = _make_state()
    control = TrainerControl()
    cb.on_train_begin(args, state, control, model=nn.Linear(4, 2))

    pid_file = tmp_path / "run" / "trainer.pid"
    assert pid_file.exists()
    assert (tmp_path / "run" / "checkpoints").is_dir()
    cb.on_train_end(args, state, control)
    assert not pid_file.exists()


def test_on_train_begin_resumed_when_global_step_nonzero(tmp_path: Path) -> None:
    cb = LensmlTrainerCallback(run_id="r", run_dir=tmp_path / "run", publisher=_make_publisher())
    args = _make_args(tmp_path)
    state = _make_state(global_step=42)
    control = TrainerControl()
    cb.on_train_begin(args, state, control, model=nn.Linear(4, 2))
    assert cb.resumed is True
    assert cb.step == 42
    assert cb.next_batch_index == 42
    cb.on_train_end(args, state, control)


def test_enable_checkpoint_false_skips_filesystem(tmp_path: Path) -> None:
    cb = LensmlTrainerCallback(
        run_id="r",
        run_dir=tmp_path / "run",
        enable_checkpoint=False,
        publisher=_make_publisher(),
    )
    args = _make_args(tmp_path)
    state = _make_state()
    control = TrainerControl()
    cb.on_train_begin(args, state, control, model=nn.Linear(4, 2))
    assert cb._signal is None
    assert not (tmp_path / "run" / "trainer.pid").exists()
    cb.on_train_end(args, state, control)


def test_on_step_end_syncs_step_from_state(tmp_path: Path) -> None:
    cb = LensmlTrainerCallback(run_id="r", run_dir=tmp_path / "run", publisher=_make_publisher())
    args = _make_args(tmp_path)
    control = TrainerControl()
    cb.on_train_begin(args, _make_state(), control, model=nn.Linear(4, 2))

    state = _make_state(global_step=7)
    cb.on_step_end(args, state, control)
    assert cb.step == 7
    assert cb.next_batch_index == 7
    cb.on_train_end(args, state, control)


def test_pre_optimizer_step_with_signal_sets_save_and_stop(tmp_path: Path) -> None:
    cb = LensmlTrainerCallback(run_id="r", run_dir=tmp_path / "run", publisher=_make_publisher())
    args = _make_args(tmp_path)
    control = TrainerControl()
    cb.on_train_begin(args, _make_state(), control, model=nn.Linear(4, 2))
    assert cb._signal is not None
    cb._signal.trigger()

    state = _make_state(global_step=3)
    cb.on_pre_optimizer_step(args, state, control)

    assert cb.stop_requested is True
    assert control.should_save is True
    assert control.should_training_stop is True
    # on_train_end will sys.exit because stop_requested is True
    with pytest.raises(SystemExit):
        cb.on_train_end(args, state, control)


def test_pre_optimizer_step_without_signal_keeps_control_unchanged(
    tmp_path: Path,
) -> None:
    cb = LensmlTrainerCallback(run_id="r", run_dir=tmp_path / "run", publisher=_make_publisher())
    args = _make_args(tmp_path)
    control = TrainerControl()
    cb.on_train_begin(args, _make_state(), control, model=nn.Linear(4, 2))

    state = _make_state(global_step=1)
    cb.on_pre_optimizer_step(args, state, control)

    assert cb.stop_requested is False
    assert control.should_training_stop is False
    cb.on_train_end(args, state, control)


def test_on_save_records_checkpoint_path(tmp_path: Path) -> None:
    cb = LensmlTrainerCallback(run_id="r", run_dir=tmp_path / "run", publisher=_make_publisher())
    args = _make_args(tmp_path)
    state = _make_state(global_step=10)
    control = TrainerControl()

    cb.on_save(args, state, control)

    assert cb.last_checkpoint_path == Path(args.output_dir) / "checkpoint-10"


def test_on_train_end_exits_when_stop_requested(tmp_path: Path) -> None:
    cb = LensmlTrainerCallback(run_id="r", run_dir=tmp_path / "run", publisher=_make_publisher())
    args = _make_args(tmp_path)
    control = TrainerControl()
    cb.on_train_begin(args, _make_state(), control, model=nn.Linear(4, 2))
    assert cb._signal is not None
    cb._signal.trigger()
    cb.on_pre_optimizer_step(args, _make_state(global_step=1), control)

    with pytest.raises(SystemExit) as exc:
        cb.on_train_end(args, _make_state(global_step=1), control)
    assert exc.value.code == EXIT_CHECKPOINTED


def test_on_train_end_no_exit_when_no_stop_requested(tmp_path: Path) -> None:
    cb = LensmlTrainerCallback(run_id="r", run_dir=tmp_path / "run", publisher=_make_publisher())
    args = _make_args(tmp_path)
    control = TrainerControl()
    cb.on_train_begin(args, _make_state(), control, model=nn.Linear(4, 2))
    # Should NOT raise SystemExit
    cb.on_train_end(args, _make_state(global_step=3), control)


# ---- end-to-end via a real Trainer ----


class _ToyDataset(Dataset):
    def __init__(self, n: int = 16) -> None:
        self.x = torch.randn(n, 4)
        self.y = torch.randint(0, 2, (n,))

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {"x": self.x[i], "labels": self.y[i]}


class _ToyModel(nn.Module):
    """Minimal model compatible with HF Trainer.

    Trainer expects ``forward(**inputs)`` to return either a tuple
    starting with the loss or a ``ModelOutput`` whose first attr is
    ``loss``. We return a dict, which Trainer accepts.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)
        self.relu = nn.ReLU()

    def forward(
        self, x: torch.Tensor, labels: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        logits = self.fc(self.relu(x))
        out: dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            out["loss"] = nn.functional.cross_entropy(logits, labels)
        return out


def _toy_collator(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "x": torch.stack([b["x"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


def _make_trainer_args(tmp_path: Path, max_steps: int = 5) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(tmp_path / "out"),
        max_steps=max_steps,
        per_device_train_batch_size=4,
        save_strategy="no",  # we'll let the callback force saves on stop
        logging_strategy="no",
        report_to="none",
        use_cpu=True,
        dataloader_num_workers=0,
        seed=42,
    )


def test_end_to_end_step_counter_advances(tmp_path: Path) -> None:
    cb = LensmlTrainerCallback(
        run_id="r-e2e", run_dir=tmp_path / "run", publisher=_make_publisher()
    )
    trainer = Trainer(
        model=_ToyModel(),
        args=_make_trainer_args(tmp_path, max_steps=3),
        train_dataset=_ToyDataset(),
        data_collator=_toy_collator,
        callbacks=[cb],
    )
    trainer.train()
    assert cb.step == 3


def test_end_to_end_stop_signal_triggers_save_and_exit(tmp_path: Path) -> None:
    """Pre-trigger the stop signal; the next pre_optimizer_step should
    request a Trainer-driven save + stop, then on_train_end exits with
    EXIT_CHECKPOINTED."""
    cb = LensmlTrainerCallback(
        run_id="r-stop", run_dir=tmp_path / "run", publisher=_make_publisher()
    )

    class _Trigger(transformers.TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            assert cb._signal is not None
            cb._signal.trigger()
            return control

    trainer = Trainer(
        model=_ToyModel(),
        args=_make_trainer_args(tmp_path, max_steps=10),
        train_dataset=_ToyDataset(),
        data_collator=_toy_collator,
        callbacks=[cb, _Trigger()],
    )

    with pytest.raises(SystemExit) as exc:
        trainer.train()
    assert exc.value.code == EXIT_CHECKPOINTED

    assert cb.stop_requested is True
    # Trainer should have written a checkpoint-<step>/ dir under output_dir
    out_dir = tmp_path / "out"
    ckpts = [p for p in out_dir.iterdir() if p.name.startswith("checkpoint-")]
    assert ckpts, f"no checkpoint dir found under {out_dir}"
