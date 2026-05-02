"""Unit tests for lensml.server.supervisor."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from lensml.server.supervisor import (
    is_alive,
    read_pid_file,
    request_stop,
    spawn_resumed_trainer,
)


def test_read_pid_file_round_trip(tmp_path: Path) -> None:
    f = tmp_path / "trainer.pid"
    f.write_text("12345\n", encoding="utf-8")
    assert read_pid_file(f) == 12345


def test_read_pid_file_missing_returns_none(tmp_path: Path) -> None:
    assert read_pid_file(tmp_path / "missing.pid") is None


def test_read_pid_file_invalid_returns_none(tmp_path: Path) -> None:
    f = tmp_path / "bad.pid"
    f.write_text("not a number", encoding="utf-8")
    assert read_pid_file(f) is None


def test_is_alive_for_self() -> None:
    assert is_alive(os.getpid()) is True


def test_is_alive_for_dead_pid() -> None:
    fake = 2**30
    assert is_alive(fake) is False


def test_request_stop_no_pid_no_flag_returns_unsignaled() -> None:
    out = request_stop(pid=None, stop_flag_path=None)
    assert out.signaled is False
    assert out.reason


def test_request_stop_only_flag_marks_signaled(tmp_path: Path) -> None:
    flag = tmp_path / "STOP"
    out = request_stop(pid=None, stop_flag_path=flag, timeout=0.1)
    assert out.signaled is True
    assert flag.exists()
    # No PID → can't poll exit
    assert out.exited is False


@pytest.mark.skipif(
    not hasattr(__import__("signal"), "SIGUSR1"),
    reason="Unix-only — request_stop polls is_alive after SIGUSR1",
)
def test_request_stop_with_pid_signals_and_polls(tmp_path: Path) -> None:
    """Mock signal_pid + is_alive to verify the poll loop logic without a
    real subprocess."""
    flag = tmp_path / "STOP"
    with (
        patch("lensml.server.supervisor.signal_pid", return_value=True) as mock_sig,
        patch("lensml.server.supervisor.is_alive", side_effect=[True, True, False]) as mock_alive,
    ):
        out = request_stop(pid=12345, stop_flag_path=flag, timeout=2.0, poll_interval=0.01)
    assert out.signaled is True
    assert out.exited is True
    mock_sig.assert_called_once_with(12345)
    assert mock_alive.call_count >= 1


@pytest.mark.skipif(
    not hasattr(__import__("signal"), "SIGUSR1"),
    reason="Unix-only — request_stop's timeout path requires PID polling",
)
def test_request_stop_timeout_when_alive_too_long() -> None:
    with (
        patch("lensml.server.supervisor.signal_pid", return_value=True),
        patch("lensml.server.supervisor.is_alive", return_value=True),
    ):
        t0 = time.time()
        out = request_stop(pid=99999, stop_flag_path=None, timeout=0.05, poll_interval=0.01)
        assert time.time() - t0 < 1.0  # well under any reasonable bound
    assert out.signaled is True
    assert out.exited is False
    assert "still alive" in out.reason


def _emit_envvar_script(out_path: Path, env_name: str) -> str:
    """Tiny script that writes the value of ``env_name`` to ``out_path``.
    Uses json.dumps to escape the path on Windows."""
    import json

    return (
        "import os\n"
        f"with open({json.dumps(str(out_path))}, 'w') as f:\n"
        f"    f.write(os.environ.get({json.dumps(env_name)}, ''))\n"
    )


def test_spawn_resumed_trainer_passes_env_var(tmp_path: Path) -> None:
    """Use a tiny inline script to confirm LENSML_RESUME is propagated."""
    script = tmp_path / "child.py"
    out_path = tmp_path / "out.txt"
    script.write_text(_emit_envvar_script(out_path, "LENSML_RESUME"), encoding="utf-8")
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_text("dummy", encoding="utf-8")

    spawned = spawn_resumed_trainer(
        cmd=[str(script)],
        resume_checkpoint=ckpt,
        python_executable=sys.executable,
    )
    spawned.process.wait(timeout=10)
    assert out_path.read_text(encoding="utf-8") == str(ckpt)


def test_spawn_resumed_trainer_extra_env(tmp_path: Path) -> None:
    script = tmp_path / "child.py"
    out_path = tmp_path / "out.txt"
    script.write_text(_emit_envvar_script(out_path, "LENSML_TEST_EXTRA"), encoding="utf-8")
    spawned = spawn_resumed_trainer(
        cmd=[str(script)],
        resume_checkpoint=tmp_path / "fake.pt",
        extra_env={"LENSML_TEST_EXTRA": "hello"},
        python_executable=sys.executable,
    )
    spawned.process.wait(timeout=10)
    assert out_path.read_text(encoding="utf-8") == "hello"
