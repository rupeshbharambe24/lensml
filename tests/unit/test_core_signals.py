"""Unit tests for lensml.core.signals."""

from __future__ import annotations

import os
import signal
from pathlib import Path

import pytest

from lensml.core.signals import (
    EXIT_CHECKPOINTED,
    CheckpointSignal,
    setup_determinism,
    signal_pid,
    touch_stop_flag,
)


def test_exit_code_constant() -> None:
    assert EXIT_CHECKPOINTED == 99


def test_should_stop_starts_false() -> None:
    sig = CheckpointSignal()
    assert sig.should_stop is False


def test_trigger_sets_should_stop() -> None:
    sig = CheckpointSignal()
    sig.trigger()
    assert sig.should_stop is True


def test_reset_clears_event_and_flag(tmp_path: Path) -> None:
    flag = tmp_path / "STOP"
    flag.touch()
    sig = CheckpointSignal(stop_flag_path=flag)
    assert sig.should_stop is True
    sig.reset()
    assert sig.should_stop is False
    assert not flag.exists()


def test_stop_flag_file_triggers_stop(tmp_path: Path) -> None:
    flag = tmp_path / "STOP"
    sig = CheckpointSignal(stop_flag_path=flag)
    assert sig.should_stop is False
    flag.touch()
    assert sig.should_stop is True


def test_install_uninstall_cycle() -> None:
    """On Unix this hooks SIGUSR1; on Windows it's a no-op. Either way,
    install/uninstall must not raise."""
    sig = CheckpointSignal()
    sig.install()
    sig.uninstall()


@pytest.mark.skipif(not hasattr(signal, "SIGUSR1"), reason="SIGUSR1 not available (Windows)")
def test_sigusr1_handler_sets_event() -> None:
    """On Unix, sending SIGUSR1 to ourselves must flip should_stop."""
    sig = CheckpointSignal()
    sig.install()
    try:
        os.kill(os.getpid(), signal.SIGUSR1)
        # Brief loop in case signal delivery is slightly async
        for _ in range(50):
            if sig.should_stop:
                break
            import time as _time

            _time.sleep(0.005)
        assert sig.should_stop is True
    finally:
        sig.uninstall()


def test_touch_stop_flag(tmp_path: Path) -> None:
    flag = tmp_path / "nested" / "STOP"
    out = touch_stop_flag(flag)
    assert flag.exists()
    assert out == flag


def test_signal_pid_unsupported_on_windows_returns_false() -> None:
    """On Windows (no SIGUSR1), signal_pid returns False without raising."""
    if hasattr(signal, "SIGUSR1"):
        pytest.skip("Unix has SIGUSR1; this asserts the Windows fallback only.")
    assert signal_pid(os.getpid()) is False


def test_signal_pid_to_dead_process_returns_false() -> None:
    """Sending to a definitely-not-running PID returns False, doesn't raise."""
    # PID 1 might exist; pick a high number that almost certainly doesn't.
    fake = 2**30
    result = signal_pid(fake)
    # On Unix returns False (ProcessLookupError caught); on Windows returns
    # False because SIGUSR1 doesn't exist. Either way: False, no raise.
    assert result is False


def test_setup_determinism_does_not_raise() -> None:
    """Smoke test — applies the deterministic recipe from spike 0.6."""
    setup_determinism(seed=123)
    # CUBLAS env should be set (or already set)
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG", "").startswith(":")
