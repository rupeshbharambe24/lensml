"""Cross-platform "please checkpoint and exit" signal for the trainer.

Plan §3.1 / §8 Day 13. Spike 0.6 validated the underlying flow:

    server (after applying a fix)
        → trigger CheckpointSignal on the trainer
            → trainer's optimizer.step wrapper sees should_stop
                → save_checkpoint() + sys.exit(99)
                    → supervisor sees exit-code 99
                        → spawn new trainer subprocess with LENSML_RESUME=...

Two mechanisms wired together:

    1. Unix: SIGUSR1 handler. The signal is delivered async; the handler
       sets a threading.Event that the trainer polls each step.
    2. Windows (no SIGUSR1): a poll for a stop-flag file. The trainer
       checks the flag each step. The server `touch`es the flag.

The same `should_stop` property covers both — callers don't branch.
"""

from __future__ import annotations

import logging
import os
import signal
import threading
from pathlib import Path

log = logging.getLogger(__name__)

EXIT_CHECKPOINTED = 99


class CheckpointSignal:
    """Trainer-side flag set by either SIGUSR1 (Unix) or a stop file (any OS).

    Usage in the trainer (typically wired by ``lensml.watch()``):

        sig = CheckpointSignal(stop_flag_path=...)
        sig.install()
        for batch in dataloader:
            ...
            if sig.should_stop:
                save_checkpoint(...)
                sys.exit(EXIT_CHECKPOINTED)
            optimizer.step()

    Server-side: tell the trainer to stop via either ``signal_pid(pid)``
    (Unix) or ``touch_stop_flag(path)`` (any OS).
    """

    __slots__ = ("_event", "_previous_handler", "_stop_flag_path")

    def __init__(self, *, stop_flag_path: Path | None = None) -> None:
        self._event = threading.Event()
        self._stop_flag_path: Path | None = stop_flag_path
        self._previous_handler: object | None = None

    @property
    def should_stop(self) -> bool:
        if self._event.is_set():
            return True
        if self._stop_flag_path is not None and self._stop_flag_path.exists():
            self._event.set()
            return True
        return False

    def trigger(self) -> None:
        """Set the stop event programmatically (tests / in-process tools)."""
        self._event.set()

    def reset(self) -> None:
        """Clear the stop event AND remove the flag file if present.

        Used by the resumed trainer right after restoring from a checkpoint
        so a leftover STOP file from the previous session doesn't fire
        immediately.
        """
        self._event.clear()
        if self._stop_flag_path is not None and self._stop_flag_path.exists():
            try:
                self._stop_flag_path.unlink()
            except OSError:
                log.warning("could not remove stale stop flag %s", self._stop_flag_path)

    def install(self) -> None:
        """Install the SIGUSR1 handler (Unix only). No-op on Windows."""
        sigusr1 = getattr(signal, "SIGUSR1", None)
        if sigusr1 is None:
            return
        try:
            self._previous_handler = signal.signal(sigusr1, self._handle)
        except (ValueError, OSError):
            log.debug("SIGUSR1 install skipped (not main thread)")
            self._previous_handler = None

    def uninstall(self) -> None:
        sigusr1 = getattr(signal, "SIGUSR1", None)
        if sigusr1 is None:
            return
        if self._previous_handler is None:
            return
        import contextlib as _ctx

        with _ctx.suppress(ValueError, OSError):
            signal.signal(sigusr1, self._previous_handler)
        self._previous_handler = None

    def _handle(self, _signum: int, _frame: object) -> None:
        """Async-signal-safe: just set the Event, do no I/O here."""
        self._event.set()


# ---- server-side helpers (call these from outside the trainer process) ----


def signal_pid(pid: int) -> bool:
    """Send SIGUSR1 to the trainer process.

    Returns True if the signal was sent, False if not supported (Windows)
    or if the PID has already exited. Never raises.
    """
    sigusr1 = getattr(signal, "SIGUSR1", None)
    if sigusr1 is None:
        return False
    try:
        os.kill(pid, sigusr1)
    except (ProcessLookupError, OSError) as e:
        log.warning("signal_pid(%d) failed: %s", pid, e)
        return False
    return True


def touch_stop_flag(path: Path) -> Path:
    """Cross-platform stop signal: just create the flag file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def setup_determinism(seed: int = 42) -> None:
    """The deterministic-mode recipe from spike 0.6.

    Must run BEFORE the trainer's first model construction or DataLoader
    creation. ``lensml.watch()`` calls this automatically when invoked
    with ``deterministic=True``; standalone trainers can call it directly.
    """
    import random as _random

    import numpy as _np
    import torch as _torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    _random.seed(seed)
    _np.random.seed(seed)
    _torch.manual_seed(seed)
    if _torch.cuda.is_available():
        _torch.cuda.manual_seed_all(seed)
    _torch.backends.cudnn.deterministic = True
    _torch.backends.cudnn.benchmark = False
    try:
        _torch.use_deterministic_algorithms(True, warn_only=True)
    except RuntimeError as e:
        log.debug("use_deterministic_algorithms unavailable: %s", e)
    _torch.set_num_threads(1)
