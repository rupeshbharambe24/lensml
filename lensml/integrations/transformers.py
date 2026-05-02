"""HuggingFace Trainer integration — `LensmlTrainerCallback`.

Plan refs:
    §2 — HF ``TrainerCallback`` is read-only (cannot mutate the loop).
         ``TrainerControl.should_training_stop = True`` is the supported
         escape hatch. lensml's pattern: callback signals stop → external
         orchestrator applies the fix → restart Trainer with
         ``resume_from_checkpoint``.
    §8 Day 16 — implementation; test on DistilBERT/SST-2 (Colab T4).
    §11 risk #1 — high probability that subprocess restart with full
         state continuity is harder than expected; fallback ships HF as
         "callback mode only, restart is manual" and documents the gap.

Architecture (parallel to ``WatchHandle`` and ``LensmlCallback``):

    - ``on_train_begin`` installs forward hooks on the model passed via
      kwargs and sets up the run_dir / PID file / CheckpointSignal.
      If ``state.global_step`` is already nonzero we treat it as a
      Trainer-managed resume (``resume_from_checkpoint`` was passed).
    - ``on_pre_optimizer_step`` is the pre-step capture point. We
      publish telemetry every ``publish_every`` steps and check the
      stop signal. If stopping: set ``control.should_save = True`` and
      ``control.should_training_stop = True``. The Trainer will write a
      ``checkpoint-<step>/`` directory on its next ``on_save`` and exit
      ``train()`` cleanly.
    - ``on_step_end`` syncs our step counter from ``state.global_step``.
    - ``on_save`` records the path of the checkpoint the Trainer just
      wrote (so the supervisor can ``LENSML_RESUME=<path>`` it).
    - ``on_train_end`` tears down hooks + PID file + publisher. If the
      stop was lensml-triggered we ``sys.exit(EXIT_CHECKPOINTED)`` so
      the supervisor sees the sentinel exit code.

Resume contract: pass ``resume_from_checkpoint=lensml_resume_ckpt()``
to ``trainer.train`` so the Trainer's native restore drives the load.
"""

from __future__ import annotations

import contextlib
import logging
import os
import secrets
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lensml.core.hooks import TelemetryBuffer, install_capture_hooks, remove_hooks
from lensml.core.signals import EXIT_CHECKPOINTED, CheckpointSignal
from lensml.core.telemetry import activation_stats, global_grad_l2
from lensml.core.transport import JSONLPublisher, ZMQPublisher, make_publisher

if TYPE_CHECKING:
    import torch.optim
    from torch.utils.hooks import RemovableHandle
    from transformers import (
        TrainerControl,
        TrainerState,
        TrainingArguments,
    )

log = logging.getLogger(__name__)

DEFAULT_PUBLISH_EVERY = 5
DEFAULT_RUN_ROOT = Path.home() / ".lensml" / "runs"
RESUME_ENV_VAR = "LENSML_RESUME"


def _generate_run_id() -> str:
    return f"run-{secrets.token_hex(4)}"


def lensml_resume_ckpt() -> str | None:
    """Helper: return the value of ``LENSML_RESUME`` (or None).

    Pass to ``trainer.train(resume_from_checkpoint=lensml_resume_ckpt())``
    so the HF Trainer resumes the run that the supervisor checkpointed.
    """
    val = os.environ.get(RESUME_ENV_VAR)
    return val if val else None


def _make_callback_base() -> type:
    """Import HF's TrainerCallback at call time so this module imports
    cleanly without transformers installed (it's an optional dep)."""
    try:
        from transformers import TrainerCallback as _TrainerCallback
    except ImportError as e:  # pragma: no cover - exercised only without extra
        raise ImportError(
            "lensml.integrations.transformers requires transformers. "
            "Install with: pip install lensml[transformers]"
        ) from e
    return _TrainerCallback


_CallbackBase = _make_callback_base()


class LensmlTrainerCallback(_CallbackBase):  # type: ignore[misc, valid-type]
    """HuggingFace Trainer callback. Use with ``Trainer(callbacks=[LensmlTrainerCallback()])``.

    Mirrors the lifecycle of ``LensmlCallback`` (Lightning) and
    ``WatchHandle`` (raw PyTorch) but adapted to HF Trainer's
    read-only callback contract: stop is requested via
    ``control.should_training_stop = True`` rather than by raising.

    Attributes:
        run_id:           short unique id for this training run.
        step:             optimizer steps completed (kept in sync with
                          ``state.global_step``).
        next_batch_index: position in the deterministic dataset; same
                          value as ``step`` for HF since Trainer
                          increments globally.
        resumed:          True if ``state.global_step > 0`` at
                          ``on_train_begin`` (Trainer was started with
                          ``resume_from_checkpoint``).
        run_dir:          per-run directory under ~/.lensml/runs/<id>/.
        last_checkpoint_path: path of the most recent
                          ``checkpoint-<step>`` directory the Trainer
                          wrote — useful for ``LENSML_RESUME``.
        stop_requested:   True after the callback set
                          ``should_training_stop`` due to a lensml stop
                          signal (vs. user-driven stop).
    """

    def __init__(
        self,
        *,
        run_id: str | None = None,
        transport: str | None = None,
        publisher: JSONLPublisher | ZMQPublisher | None = None,
        publish_every: int = DEFAULT_PUBLISH_EVERY,
        run_dir: Path | None = None,
        enable_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        self.run_id = run_id or _generate_run_id()
        self.step = 0
        self.next_batch_index = 0
        self.resumed = False
        self.publish_every = publish_every
        self.enable_checkpoint = enable_checkpoint
        self.run_dir = run_dir if run_dir is not None else DEFAULT_RUN_ROOT / self.run_id
        self.last_checkpoint_path: Path | None = None
        self.stop_requested = False

        self._transport = transport
        self._publisher = publisher
        self._buffer = TelemetryBuffer()
        self._hook_handles: list[RemovableHandle] = []
        self._signal: CheckpointSignal | None = None
        self._pid_file: Path | None = None
        self._stop_flag_path: Path | None = None
        self._model: Any = None
        self._closed = False

    # ---- HF Trainer hooks ----

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        if self._publisher is None:
            self._publisher = make_publisher(self._transport)

        self._model = kwargs.get("model")
        if self._model is not None:
            self._hook_handles = install_capture_hooks(self._model, self._buffer)

        if self.enable_checkpoint:
            self._setup_run_dir()
            self._install_signal()

        # If Trainer is resuming (resume_from_checkpoint was set), state
        # already has global_step > 0 by this hook.
        if state.global_step > 0:
            self.step = state.global_step
            self.next_batch_index = state.global_step
            self.resumed = True

        self._publish_safely(
            {
                "run_id": self.run_id,
                "kind": "run_started",
                "ts": time.time(),
                "resumed": self.resumed,
                "step": self.step,
                "pid": os.getpid(),
            }
        )
        return control

    def on_pre_optimizer_step(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        if self.enable_checkpoint and self._signal is not None and self._signal.should_stop:
            # HF Trainer is read-only — request a save + stop and let the
            # Trainer drive the actual checkpoint write.
            self.stop_requested = True
            control.should_save = True
            control.should_training_stop = True
            log.info(
                "stop signal received; requesting Trainer save + stop at step %d",
                state.global_step,
            )
            return control

        if self.step % self.publish_every == 0:
            self._publish_step(kwargs.get("optimizer"))
        return control

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        # Trainer increments global_step on each effective optimizer step
        # (gradient accumulation already accounted for).
        self.step = int(state.global_step)
        self.next_batch_index = int(state.global_step)
        return control

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        # Trainer just wrote (or is about to write) checkpoint-<step>/.
        ckpt_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        self.last_checkpoint_path = ckpt_dir
        if self.stop_requested:
            self._publish_safely(
                {
                    "run_id": self.run_id,
                    "kind": "checkpointed",
                    "ts": time.time(),
                    "step": int(state.global_step),
                    "checkpoint_path": str(ckpt_dir),
                }
            )
        return control

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        self._teardown()
        if self.stop_requested:
            sys.exit(EXIT_CHECKPOINTED)
        return control

    # ---- internals ----

    def _setup_run_dir(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)
        self._pid_file = self.run_dir / "trainer.pid"
        try:
            self._pid_file.write_text(str(os.getpid()), encoding="utf-8")
        except OSError as e:
            log.warning("could not write PID file %s: %s", self._pid_file, e)
            self._pid_file = None
        self._stop_flag_path = self.run_dir / "STOP"

    def _install_signal(self) -> None:
        self._signal = CheckpointSignal(stop_flag_path=self._stop_flag_path)
        self._signal.reset()
        self._signal.install()

    def _publish_step(self, optimizer: torch.optim.Optimizer | None) -> None:
        try:
            act = activation_stats(self._buffer)
            grad_l2 = global_grad_l2(self._model) if self._model is not None else 0.0
        except Exception:
            return
        lr: float | None = None
        if optimizer is not None and getattr(optimizer, "param_groups", None):
            lr = float(optimizer.param_groups[0].get("lr", 0.0))
        self._publish_safely(
            {
                "run_id": self.run_id,
                "kind": "metric",
                "step": self.step,
                "ts": time.time(),
                "grad_l2": grad_l2,
                "lr": lr,
                "activations": act,
            }
        )

    def _publish_safely(self, event: dict[str, Any]) -> None:
        if self._publisher is None:
            return
        with contextlib.suppress(Exception):
            self._publisher.publish(event)

    def _teardown(self) -> None:
        if self._closed:
            return
        self._closed = True

        remove_hooks(self._hook_handles)
        self._buffer.clear()

        if self._signal is not None:
            self._signal.uninstall()

        if self._pid_file is not None and self._pid_file.exists():
            with contextlib.suppress(OSError):
                self._pid_file.unlink()

        self._publish_safely(
            {
                "run_id": self.run_id,
                "kind": "run_ended",
                "ts": time.time(),
                "step": self.step,
            }
        )
        if self._publisher is not None:
            with contextlib.suppress(Exception):
                self._publisher.close()


__all__ = [
    "RESUME_ENV_VAR",
    "LensmlTrainerCallback",
    "lensml_resume_ckpt",
]
