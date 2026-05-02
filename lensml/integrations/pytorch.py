"""Raw PyTorch integration — the `lensml.watch()` one-line API.

Plan refs:
    §1 — `lensml.watch(model, optimizer, dataloader)` installs hooks, streams
         metrics to the local FastAPI server, monitors for pathologies.
    §8 Day 5 — Monkey-patch `optimizer.step` for pre-step gradient inspection.
               RemovableHandle teardown via context manager or explicit
               `lensml.unwatch()`.
    §8 Day 13 — checkpoint resume: `lensml.watch()` checks `LENSML_RESUME`
                env var on init.

Architecture (validated by spike 0.4 + step 1.2/1.3 + step 2.6):
    - Forward hooks on activation modules capture output references in a
      `TelemetryBuffer` (effectively zero hot-path cost).
    - `optimizer.step` is wrapped: BEFORE calling the original step, we
      check the `CheckpointSignal` (SIGUSR1 on Unix / stop-flag on Windows)
      and read `param.grad` (still valid pre-step) to compute summary stats
      every PUBLISH_EVERY-th call.
    - Stats are published via the transport layer (ZMQ PUB by default,
      JSONL fallback).
    - On stop signal: save_checkpoint then sys.exit(99). Server's supervisor
      then spawns a new trainer process with LENSML_RESUME=<ckpt_path> to
      pick up exactly where we left off.
    - On startup with LENSML_RESUME set: load_checkpoint into model +
      optimizer, restore step counter and next_batch_index.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import os
import secrets
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lensml.core.hooks import (
    TelemetryBuffer,
    install_capture_hooks,
    remove_hooks,
)
from lensml.core.signals import EXIT_CHECKPOINTED, CheckpointSignal
from lensml.core.telemetry import activation_stats, global_grad_l2
from lensml.core.transport import (
    JSONLPublisher,
    ZMQPublisher,
    make_publisher,
)
from lensml.server.checkpoint import load_checkpoint, save_checkpoint

if TYPE_CHECKING:
    import torch.nn as nn
    import torch.optim
    from torch.utils.data import DataLoader
    from torch.utils.hooks import RemovableHandle


log = logging.getLogger(__name__)

DEFAULT_PUBLISH_EVERY = 5  # plan §3.1 tick rate (~10-50 Hz envelope)
DEFAULT_RUN_ROOT = Path.home() / ".lensml" / "runs"
RESUME_ENV_VAR = "LENSML_RESUME"


def _generate_run_id() -> str:
    """Short random run id matching plan §7.1's example format ('run-a8f3...')."""
    return f"run-{secrets.token_hex(4)}"


def _current_lr(optimizer: torch.optim.Optimizer) -> float | None:
    """First param group's lr, or None if not set."""
    if not optimizer.param_groups:
        return None
    return float(optimizer.param_groups[0].get("lr", 0.0))


class WatchHandle:
    """Lifecycle handle returned by `lensml.watch()`.

    Owns the hooks, the optimizer.step monkey-patch, the publisher socket,
    and the checkpoint signal handler. Use as a context manager or call
    `.unwatch()` explicitly.

    Attributes:
        run_id:           short unique id for this training run.
        step:             optimizer steps completed since the original
                          start of training (i.e. continues across resume).
        next_batch_index: position in the deterministic dataset to resume
                          from. Trainers that want bit-exact resume should
                          fast-forward their dataloader to this index after
                          ``watch()`` returns. (See spike 0.6 for the
                          recipe.)
        resumed:          True if this handle was constructed from a
                          checkpoint (LENSML_RESUME was set).
        run_dir:          per-run directory under ~/.lensml/runs/<run_id>/
                          where the trainer.pid + checkpoints + STOP flag
                          live.
    """

    def __init__(
        self,
        *,
        run_id: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        publisher: JSONLPublisher | ZMQPublisher,
        publish_every: int = DEFAULT_PUBLISH_EVERY,
        run_dir: Path | None = None,
        enable_checkpoint: bool = True,
        scheduler: Any | None = None,
    ) -> None:
        self.run_id = run_id
        self.step = 0
        self.next_batch_index = 0
        self.resumed = False
        self.publish_every = publish_every
        self.enable_checkpoint = enable_checkpoint
        self.run_dir = run_dir if run_dir is not None else DEFAULT_RUN_ROOT / run_id

        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._publisher = publisher
        self._buffer = TelemetryBuffer()
        self._hook_handles: list[RemovableHandle] = []
        self._original_step: Any | None = None
        self._closed = False

        # Checkpoint / signal infrastructure (filled in _install).
        self._signal: CheckpointSignal | None = None
        self._pid_file: Path | None = None
        self._stop_flag_path: Path | None = None

    # ---- setup / teardown ----

    def _install(self) -> None:
        """Internal: install forward hooks, wrap optimizer.step, set up
        the checkpoint signal handler + PID file, and load from
        LENSML_RESUME if set.

        Idempotent — skips if already installed.
        """
        if self._original_step is not None:
            return

        if self.enable_checkpoint:
            self._setup_run_dir()
            self._maybe_resume()
            self._install_signal()

        self._hook_handles = install_capture_hooks(self._model, self._buffer)

        original = self._optimizer.step
        self._original_step = original

        @functools.wraps(original)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            # Stop check BEFORE the step — gives a clean checkpoint with
            # state matching the published telemetry.
            if self.enable_checkpoint and self._signal is not None and self._signal.should_stop:
                self._handle_stop()
                # _handle_stop calls sys.exit; this return is unreachable
                # but appeases static analysis.
                return None

            # Capture telemetry BEFORE the step — grads are still valid here.
            if self.step % self.publish_every == 0:
                self._publish_step()
            result = original(*args, **kwargs)
            self.step += 1
            self.next_batch_index += 1
            return result

        # Setting an instance attribute on the optimizer shadows the class-
        # level bound method until we del it again in unwatch().
        self._optimizer.step = wrapped  # type: ignore[method-assign]

        # Initial run-start event so the server's listing populates.
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

    def unwatch(self) -> None:
        """Remove hooks, restore optimizer.step, close publisher,
        uninstall signal handler, remove PID file.

        Idempotent — calling twice is a no-op.
        """
        if self._closed:
            return
        self._closed = True

        if self._original_step is not None:
            with contextlib.suppress(AttributeError):
                del self._optimizer.step  # type: ignore[method-assign]
            self._original_step = None

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
        self._publisher.close()

    def __enter__(self) -> WatchHandle:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.unwatch()

    # ---- checkpoint / signal plumbing ----

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
        # Clear any leftover STOP file from a prior session before installing.
        self._signal.reset()
        self._signal.install()

    def _maybe_resume(self) -> None:
        path_str = os.environ.get(RESUME_ENV_VAR)
        if not path_str:
            return
        path = Path(path_str)
        if not path.exists():
            log.warning("%s set to %s but file does not exist", RESUME_ENV_VAR, path)
            return
        try:
            info = load_checkpoint(
                path,
                model=self._model,
                optimizer=self._optimizer,
                scheduler=self._scheduler,
            )
        except Exception:
            log.exception("failed to load checkpoint from %s", path)
            return
        self.step = info["step"]
        self.next_batch_index = info["next_batch_index"]
        self.resumed = True
        log.info(
            "resumed run_id=%s from %s (step=%d, next_batch=%d)",
            self.run_id,
            path,
            self.step,
            self.next_batch_index,
        )

    def _checkpoint_path(self) -> Path:
        return self.run_dir / "checkpoints" / f"step-{self.step:08d}.pt"

    def checkpoint(self, path: Path | None = None) -> Path:
        """Save a checkpoint of the current trainer state. Returns the path.

        Public API so user code can checkpoint manually (e.g. on epoch
        boundaries) without waiting for a stop signal.
        """
        path = path if path is not None else self._checkpoint_path()
        save_checkpoint(
            path,
            step=self.step,
            next_batch_index=self.next_batch_index,
            model=self._model,
            optimizer=self._optimizer,
            scheduler=self._scheduler,
            extra={
                "run_id": self.run_id,
                "ts": time.time(),
                "pid": os.getpid(),
            },
        )
        return path

    def _handle_stop(self) -> None:
        """Save a checkpoint and exit with code 99 (the "checkpointed
        cleanly" sentinel)."""
        try:
            path = self.checkpoint()
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
            self._publisher.close()
        sys.exit(EXIT_CHECKPOINTED)

    # ---- publish ----

    def log(self, **fields: Any) -> None:
        """Publish a user-supplied event tagged with the current step.

        Use for loss, accuracy, learning-rate decisions, etc. — anything
        the trainer knows that lensml's hooks don't see automatically.

        Example:
            handle.log(loss=loss.item(), accuracy=acc)
        """
        self._publish_safely(
            {
                "run_id": self.run_id,
                "kind": "metric",
                "step": self.step,
                "ts": time.time(),
                **fields,
            }
        )

    def _publish_step(self) -> None:
        """Publish auto-captured telemetry: activation stats + global grad L2."""
        try:
            act = activation_stats(self._buffer)
            grad_l2 = global_grad_l2(self._model)
        except Exception:
            return  # never crash the user's training loop
        self._publish_safely(
            {
                "run_id": self.run_id,
                "kind": "metric",
                "step": self.step,
                "ts": time.time(),
                "grad_l2": grad_l2,
                "lr": _current_lr(self._optimizer),
                "activations": act,
            }
        )

    def _publish_safely(self, event: dict[str, Any]) -> None:
        # Telemetry must never crash training — swallow any publisher error.
        with contextlib.suppress(Exception):
            self._publisher.publish(event)


def watch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader | None = None,
    *,
    run_id: str | None = None,
    transport: str | None = None,
    publisher: JSONLPublisher | ZMQPublisher | None = None,
    publish_every: int = DEFAULT_PUBLISH_EVERY,
    run_dir: Path | None = None,
    enable_checkpoint: bool = True,
    scheduler: Any | None = None,
) -> WatchHandle:
    """Install lensml supervision on a PyTorch training loop.

    Plan §1: this is THE one-line user-facing API.

    Args:
        model: any `torch.nn.Module`. Forward hooks are installed on its
            activation submodules (ReLU/GELU/etc).
        optimizer: any `torch.optim.Optimizer`. Its `.step` is wrapped to
            publish telemetry every `publish_every` steps and to honor
            stop signals.
        dataloader: accepted for API compatibility; currently unused. Will
            be instrumented for the dataloader_bottleneck detector in v0.2.
        run_id: human-readable run identifier. Auto-generated if not given.
        transport: endpoint string passed to `make_publisher` (e.g.
            ``"tcp://127.0.0.1:5555"`` or ``"jsonl"``). Ignored if
            ``publisher`` is provided.
        publisher: pre-built publisher instance. Tests use this to inject
            a `JSONLPublisher` writing to an in-memory buffer.
        publish_every: how often (in optimizer steps) the auto-captured
            telemetry is emitted. Default 5 ≈ 20 Hz on a 10 ms/step CPU
            loop, matching plan §3.1's 10–50 Hz tick envelope.
        run_dir: per-run directory for the trainer.pid + STOP flag +
            checkpoints. Defaults to ``~/.lensml/runs/<run_id>/``.
        enable_checkpoint: if True (default), set up the run dir, write
            the PID file, install the checkpoint signal handler, and
            check ``LENSML_RESUME`` to load from a checkpoint. Set to
            False in tests that don't want filesystem side effects.
        scheduler: optional learning-rate scheduler — its state_dict is
            also saved/loaded with the checkpoint.

    Returns:
        A `WatchHandle` — call `.unwatch()` or use as a context manager.
    """
    # `dataloader` is accepted for forward-compat but unused for now.
    del dataloader

    if publisher is None:
        publisher = make_publisher(transport)

    handle = WatchHandle(
        run_id=run_id or _generate_run_id(),
        model=model,
        optimizer=optimizer,
        publisher=publisher,
        publish_every=publish_every,
        run_dir=run_dir,
        enable_checkpoint=enable_checkpoint,
        scheduler=scheduler,
    )
    handle._install()
    return handle


def unwatch(handle: WatchHandle) -> None:
    """Tear down a previously-installed watch."""
    handle.unwatch()
