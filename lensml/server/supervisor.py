"""Trainer subprocess supervisor — plan §8 Day 13.

The supervisor's job (the auto-restart half of the magic moment):

    1. Track the trainer's PID (registered via PID file by lensml.watch).
    2. On user approve: send the stop signal (SIGUSR1 / stop-flag).
    3. Wait for the trainer to exit cleanly with code EXIT_CHECKPOINTED.
    4. Spawn a fresh trainer subprocess with LENSML_RESUME=<ckpt_path>
       so the new process picks up exactly where the old one left off.

For v1 we keep this minimal — start / signal / wait / spawn — and rely
on PID/flag-file rendezvous rather than parent-child IPC. Production
deployment scenario is identical: the user runs the trainer themselves
and registers it with the server via the PID file lensml.watch writes.

End-to-end orchestration tests live in `tests/integration/test_*` and
exercise the full kill+restart cycle with real subprocesses. Spike 0.6
already proved byte-identical resume; this module is the production
glue around that proof.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from lensml.core.signals import EXIT_CHECKPOINTED, signal_pid, touch_stop_flag

log = logging.getLogger(__name__)


def read_pid_file(path: Path) -> int | None:
    """Read a PID file written by lensml.watch. Returns None if missing/invalid."""
    if not path.exists():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def is_alive(pid: int) -> bool:
    """Best-effort liveness check via os.kill(pid, 0).

    Unix: ProcessLookupError → dead; EPERM (errno 1) → alive but unreachable.
    Windows: os.kill(pid, 0) for a non-existent PID raises OSError (EINVAL or
    EBADF depending on the runtime); we treat any non-EPERM OSError as dead.
    Documented limitation: rare Unix EPERM cases on processes we don't own
    will return True. Acceptable for v1 — the trainer is always our child.
    """
    import errno as _errno

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError as e:
        return getattr(e, "errno", None) == _errno.EPERM
    return True


@dataclass
class StopOutcome:
    """Outcome of asking a trainer to stop."""

    signaled: bool
    exited: bool
    exit_code: int | None = None
    elapsed: float = 0.0
    reason: str = ""


def request_stop(
    *,
    pid: int | None,
    stop_flag_path: Path | None,
    timeout: float = 30.0,
    poll_interval: float = 0.1,
) -> StopOutcome:
    """Ask the trainer to stop and wait until it exits.

    Sends SIGUSR1 (if PID available) AND touches the stop-flag file (if
    path provided), so either mechanism can win. Polls liveness until
    timeout. ``exit_code`` is None when we couldn't read it (we don't
    own the child, only see liveness).
    """
    t0 = time.time()
    signaled_any = False

    if stop_flag_path is not None:
        try:
            touch_stop_flag(stop_flag_path)
            signaled_any = True
        except OSError as e:
            log.warning("touch_stop_flag failed: %s", e)

    if pid is not None and signal_pid(pid):
        signaled_any = True

    if not signaled_any:
        return StopOutcome(
            signaled=False,
            exited=False,
            elapsed=time.time() - t0,
            reason="no PID and no stop_flag_path provided",
        )

    if pid is None:
        return StopOutcome(
            signaled=True,
            exited=False,
            elapsed=time.time() - t0,
            reason="no PID — flag set but liveness can't be polled",
        )

    while time.time() - t0 < timeout:
        if not is_alive(pid):
            return StopOutcome(
                signaled=True,
                exited=True,
                elapsed=time.time() - t0,
            )
        time.sleep(poll_interval)

    return StopOutcome(
        signaled=True,
        exited=False,
        elapsed=time.time() - t0,
        reason=f"trainer pid={pid} still alive after {timeout}s",
    )


@dataclass
class SpawnedTrainer:
    process: subprocess.Popen
    cmd: list[str]
    env: dict[str, str] = field(default_factory=dict)


def spawn_resumed_trainer(
    *,
    cmd: list[str],
    resume_checkpoint: Path,
    extra_env: dict[str, str] | None = None,
    cwd: Path | None = None,
    python_executable: str | None = None,
) -> SpawnedTrainer:
    """Spawn the trainer subprocess with ``LENSML_RESUME=<resume_checkpoint>``.

    `cmd` is the trainer command WITHOUT a leading interpreter — e.g.
    ``["train.py", "--epochs", "10"]``. The interpreter (`sys.executable`
    by default) is prepended automatically.
    """
    interp = python_executable or sys.executable
    full_cmd = [interp, *cmd]
    env = os.environ.copy()
    env["LENSML_RESUME"] = str(resume_checkpoint)
    if extra_env:
        env.update(extra_env)
    log.info("supervisor spawning %s (LENSML_RESUME=%s)", full_cmd, resume_checkpoint)
    proc = subprocess.Popen(
        full_cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
    )
    return SpawnedTrainer(process=proc, cmd=full_cmd, env=env)


__all__ = [
    "EXIT_CHECKPOINTED",
    "SpawnedTrainer",
    "StopOutcome",
    "is_alive",
    "read_pid_file",
    "request_stop",
    "spawn_resumed_trainer",
]
