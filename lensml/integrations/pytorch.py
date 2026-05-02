"""Raw PyTorch integration — the `lensml.watch()` one-line API.

Plan refs:
    §1 — `lensml.watch(model, optimizer, dataloader)` installs hooks, streams
         metrics to the local FastAPI server, monitors for pathologies.
    §8 Day 5 — Monkey-patch `optimizer.step` for pre-step gradient inspection.
               RemovableHandle teardown via context manager or explicit
               `lensml.unwatch()`.
    §8 Day 13 — checkpoint resume: `lensml.watch()` checks `LENSML_RESUME`
                env var on init (deferred to Phase 2).

Architecture (validated by spike 0.4 + step 1.2/1.3):
    - Forward hooks on activation modules capture output references in a
      `TelemetryBuffer` (effectively zero hot-path cost).
    - `optimizer.step` is wrapped: BEFORE calling the original step, we
      read `param.grad` (still valid pre-step) and compute summary stats
      every PUBLISH_EVERY-th call.
    - Stats are published via the transport layer (ZMQ PUB by default,
      JSONL fallback). The lensml server's SUB bridge picks them up and
      forwards to the dashboard via SSE.
"""

from __future__ import annotations

import contextlib
import functools
import secrets
import time
from typing import TYPE_CHECKING, Any

from lensml.core.hooks import (
    TelemetryBuffer,
    install_capture_hooks,
    remove_hooks,
)
from lensml.core.telemetry import activation_stats, global_grad_l2
from lensml.core.transport import (
    JSONLPublisher,
    ZMQPublisher,
    make_publisher,
)

if TYPE_CHECKING:
    import torch.nn as nn
    import torch.optim
    from torch.utils.data import DataLoader
    from torch.utils.hooks import RemovableHandle


DEFAULT_PUBLISH_EVERY = 5  # plan §3.1 tick rate (~10-50 Hz envelope)


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

    Owns the hooks, the optimizer.step monkey-patch, and the publisher
    socket. Use as a context manager or call `.unwatch()` explicitly.

    Attributes:
        run_id: short unique id used as the SQLite/UI run identifier.
        step:   number of completed optimizer steps since watch started.
    """

    def __init__(
        self,
        *,
        run_id: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        publisher: JSONLPublisher | ZMQPublisher,
        publish_every: int = DEFAULT_PUBLISH_EVERY,
    ) -> None:
        self.run_id = run_id
        self.step = 0
        self.publish_every = publish_every
        self._model = model
        self._optimizer = optimizer
        self._publisher = publisher
        self._buffer = TelemetryBuffer()
        self._hook_handles: list[RemovableHandle] = []
        self._original_step: Any | None = None
        self._closed = False

    # ---- setup / teardown ----

    def _install(self) -> None:
        """Internal: install forward hooks and wrap optimizer.step.

        Called by `watch()` immediately after construction. Idempotent —
        skips if already installed.
        """
        if self._original_step is not None:
            return

        self._hook_handles = install_capture_hooks(self._model, self._buffer)

        original = self._optimizer.step
        self._original_step = original

        @functools.wraps(original)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            # Capture telemetry BEFORE the step — grads are still valid here.
            if self.step % self.publish_every == 0:
                self._publish_step()
            result = original(*args, **kwargs)
            self.step += 1
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
            }
        )

    def unwatch(self) -> None:
        """Remove hooks, restore optimizer.step, close publisher.

        Idempotent — calling twice is a no-op.
        """
        if self._closed:
            return
        self._closed = True

        if self._original_step is not None:
            # Drop the instance attribute we added; falls back to the class method.
            with contextlib.suppress(AttributeError):
                del self._optimizer.step  # type: ignore[method-assign]
            self._original_step = None

        remove_hooks(self._hook_handles)
        self._buffer.clear()

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
) -> WatchHandle:
    """Install lensml supervision on a PyTorch training loop.

    Plan §1: this is THE one-line user-facing API.

    Args:
        model: any `torch.nn.Module`. Forward hooks are installed on its
            activation submodules (ReLU/GELU/etc).
        optimizer: any `torch.optim.Optimizer`. Its `.step` is wrapped to
            publish telemetry every `publish_every` steps.
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
    )
    handle._install()
    return handle


def unwatch(handle: WatchHandle) -> None:
    """Tear down a previously-installed watch."""
    handle.unwatch()
