"""PyTorch Lightning integration — `LensmlCallback`.

Plan refs:
    §2 — Lightning callbacks expose ``on_train_batch_start/end``,
         ``on_before_optimizer_step``, ``on_after_backward``,
         ``on_save_checkpoint``, ``on_load_checkpoint``.
         ``Trainer.fit(ckpt_path=...)`` resumes with full state.
    §8 Day 15 — implementation; test on CIFAR-10 + small CNN.

Architecture (parallel to ``lensml.integrations.pytorch.WatchHandle``):

    - ``on_train_start`` installs forward hooks on the LightningModule and
      sets up the run_dir / PID file / CheckpointSignal exactly the same
      way the raw PyTorch handle does.
    - ``on_before_optimizer_step`` is the pre-step capture point — grads
      are still valid here. We publish telemetry every ``publish_every``
      steps and check ``should_stop``. On stop we call
      ``trainer.save_checkpoint`` (Lightning's own .ckpt format, which
      already includes optimizer + scheduler + RNG state) then
      ``sys.exit(EXIT_CHECKPOINTED)``.
    - ``on_save_checkpoint`` injects ``run_id`` + ``step`` +
      ``next_batch_index`` into the checkpoint dict so a later resume can
      restore lensml's own counters.
    - ``on_load_checkpoint`` reads them back and flips ``resumed = True``.

Resume contract: pass ``ckpt_path=lensml_resume_ckpt()`` to
``Trainer.fit`` so Lightning's native restore drives the load. The
supervisor sets ``LENSML_RESUME`` for us; the helper just reads it.
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
    import lightning.pytorch as pl
    import torch.optim
    from torch.utils.hooks import RemovableHandle

log = logging.getLogger(__name__)

DEFAULT_PUBLISH_EVERY = 5
DEFAULT_RUN_ROOT = Path.home() / ".lensml" / "runs"
RESUME_ENV_VAR = "LENSML_RESUME"
LENSML_CKPT_KEY = "lensml"


def _generate_run_id() -> str:
    return f"run-{secrets.token_hex(4)}"


def lensml_resume_ckpt() -> str | None:
    """Helper: return the value of ``LENSML_RESUME`` (or None).

    Pass to ``Trainer.fit(ckpt_path=lensml_resume_ckpt())`` to make
    Lightning resume the run that was checkpointed when the supervisor
    sent the stop signal.
    """
    val = os.environ.get(RESUME_ENV_VAR)
    return val if val else None


def _try_get_optimizer_lr(trainer: pl.Trainer) -> float | None:
    """Best-effort: first optimizer's first param group lr."""
    optimizers = trainer.optimizers if hasattr(trainer, "optimizers") else []
    if not optimizers:
        return None
    opt = optimizers[0]
    if not opt.param_groups:
        return None
    return float(opt.param_groups[0].get("lr", 0.0))


def _make_callback_base() -> type:
    """Import Lightning's Callback at call time so the module imports
    cleanly without lightning installed (it's an optional dep)."""
    try:
        from lightning.pytorch.callbacks import Callback as _Callback
    except ImportError:  # pragma: no cover - exercised only without extra
        try:
            from pytorch_lightning.callbacks import Callback as _Callback
        except ImportError as e:
            raise ImportError(
                "lensml.integrations.lightning requires pytorch-lightning. "
                "Install with: pip install lensml[lightning]"
            ) from e
    return _Callback


_CallbackBase = _make_callback_base()


class LensmlCallback(_CallbackBase):  # type: ignore[misc, valid-type]
    """Lightning callback. Pass to ``pl.Trainer(callbacks=[LensmlCallback()])``.

    Mirrors the ``WatchHandle`` lifecycle from the raw PyTorch
    integration — same run_dir layout, same CheckpointSignal, same
    EXIT_CHECKPOINTED=99 sentinel — but bound to Lightning's hooks
    instead of monkey-patching ``optimizer.step``.

    Attributes (mostly read-only after install):
        run_id:           short unique id for this training run.
        step:             optimizer steps completed since the original
                          start of training (continues across resume).
        next_batch_index: position in the deterministic dataset to
                          resume from. (Lightning normally manages this
                          via its own dataloader-state restore, but we
                          track it for the supervisor.)
        resumed:          True when the callback was reconstructed from
                          a Lightning checkpoint that contained our
                          metadata block.
        run_dir:          per-run directory under ~/.lensml/runs/<id>/.
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

        # Publisher is built lazily so the user can pass a pre-built one
        # (tests do this to inject an in-memory JSONLPublisher).
        self._transport = transport
        self._publisher = publisher
        self._buffer = TelemetryBuffer()
        self._hook_handles: list[RemovableHandle] = []
        self._signal: CheckpointSignal | None = None
        self._pid_file: Path | None = None
        self._stop_flag_path: Path | None = None
        self._installed = False
        self._closed = False

    # ---- public API ----

    def checkpoint(self, trainer: pl.Trainer, path: Path | None = None) -> Path:
        """Save a Lightning checkpoint at the current state. Returns the path.

        Uses ``trainer.save_checkpoint`` so the resulting .ckpt is
        Lightning-native (model + optimizers + schedulers + RNG +
        callback states + our metadata block).
        """
        path = path if path is not None else self._checkpoint_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(path))
        return path

    # ---- Lightning hooks ----

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._publisher is None:
            self._publisher = make_publisher(self._transport)

        if self.enable_checkpoint:
            self._setup_run_dir()
            self._install_signal()

        self._hook_handles = install_capture_hooks(pl_module, self._buffer)
        self._installed = True

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

    def on_before_optimizer_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        # Pre-step is the right place to (a) inspect grads while they're
        # still valid and (b) intercept the stop signal so the saved
        # checkpoint matches the published telemetry.
        if self.enable_checkpoint and self._signal is not None and self._signal.should_stop:
            self._handle_stop(trainer)
            return  # _handle_stop calls sys.exit; unreachable but explicit.

        if self.step % self.publish_every == 0:
            self._publish_step(pl_module, optimizer, trainer)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self.step += 1
        self.next_batch_index += 1

    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: dict[str, Any],
    ) -> None:
        # Inject our metadata into Lightning's checkpoint dict so a
        # subsequent on_load_checkpoint can restore counters + run_id.
        checkpoint[LENSML_CKPT_KEY] = {
            "run_id": self.run_id,
            "step": self.step,
            "next_batch_index": self.next_batch_index,
            "ts": time.time(),
        }

    def on_load_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: dict[str, Any],
    ) -> None:
        meta = checkpoint.get(LENSML_CKPT_KEY)
        if not meta:
            return
        self.run_id = meta.get("run_id", self.run_id)
        self.step = int(meta.get("step", self.step))
        self.next_batch_index = int(meta.get("next_batch_index", self.next_batch_index))
        self.resumed = True
        log.info(
            "lensml callback resumed run_id=%s step=%d next_batch=%d",
            self.run_id,
            self.step,
            self.next_batch_index,
        )

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._teardown()

    # ---- internals ----

    def _checkpoint_path(self) -> Path:
        return self.run_dir / "checkpoints" / f"step-{self.step:08d}.ckpt"

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

    def _handle_stop(self, trainer: pl.Trainer) -> None:
        try:
            path = self.checkpoint(trainer)
            log.info(
                "stop signal received; checkpointed run_id=%s step=%d → %s",
                self.run_id,
                self.step,
                path,
            )
            self._publish_safely(
                {
                    "run_id": self.run_id,
                    "kind": "checkpointed",
                    "ts": time.time(),
                    "step": self.step,
                    "checkpoint_path": str(path),
                }
            )
        except Exception:
            log.exception("checkpoint save failed; exiting anyway")
        with contextlib.suppress(Exception):
            if self._publisher is not None:
                self._publisher.close()
        sys.exit(EXIT_CHECKPOINTED)

    def _publish_step(
        self,
        pl_module: pl.LightningModule,
        optimizer: torch.optim.Optimizer,
        trainer: pl.Trainer,
    ) -> None:
        try:
            act = activation_stats(self._buffer)
            grad_l2 = global_grad_l2(pl_module)
        except Exception:
            return  # never crash training on telemetry failure
        lr = None
        if optimizer.param_groups:
            lr = float(optimizer.param_groups[0].get("lr", 0.0))
        else:
            lr = _try_get_optimizer_lr(trainer)
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
    "LENSML_CKPT_KEY",
    "RESUME_ENV_VAR",
    "LensmlCallback",
    "lensml_resume_ckpt",
]
