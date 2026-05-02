"""Unit tests for lensml.server.git."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from lensml.server.git import GitError, GitRepo

# Skip the whole module if git CLI isn't available (CI runners always have it,
# but be defensive).
pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git CLI not available")


def test_is_repo_false_for_empty_dir(tmp_path: Path) -> None:
    repo = GitRepo(tmp_path)
    assert repo.is_repo() is False


def test_init_if_needed_creates_repo(tmp_path: Path) -> None:
    repo = GitRepo(tmp_path / "shadow")
    assert repo.init_if_needed() is True
    assert repo.is_repo() is True
    # Idempotent — second call doesn't re-init
    assert repo.init_if_needed() is False


def test_init_creates_seed_commit(tmp_path: Path) -> None:
    """After init, HEAD must resolve so reset_hard(HEAD~1) won't underflow
    later."""
    repo = GitRepo(tmp_path / "shadow")
    repo.init_if_needed()
    log = repo.log_oneline(5)
    assert len(log) >= 1
    assert "init shadow repo" in log[0]


def test_create_branch_and_current_branch(tmp_path: Path) -> None:
    repo = GitRepo(tmp_path / "shadow")
    repo.init_if_needed()
    assert repo.current_branch() == "main"
    repo.create_branch("lensml/session-test")
    assert repo.current_branch() == "lensml/session-test"


def test_commit_file_round_trip(tmp_path: Path) -> None:
    repo = GitRepo(tmp_path / "shadow")
    repo.init_if_needed()
    f = repo.work_dir / "train.py"
    f.write_text("x = 1\n", encoding="utf-8")
    sha = repo.commit_file(f, "first")
    assert sha
    assert len(sha) == 40  # full SHA1
    log = repo.log_oneline(5)
    assert any("first" in line for line in log)


def test_reset_hard_reverts_last_commit(tmp_path: Path) -> None:
    repo = GitRepo(tmp_path / "shadow")
    repo.init_if_needed()
    f = repo.work_dir / "train.py"
    f.write_text("x = 1\n", encoding="utf-8")
    repo.commit_file(f, "v1")
    f.write_text("x = 2\n", encoding="utf-8")
    repo.commit_file(f, "v2")
    assert f.read_text(encoding="utf-8") == "x = 2\n"

    repo.reset_hard("HEAD~1")
    assert f.read_text(encoding="utf-8") == "x = 1\n"


def test_commit_file_outside_work_dir_raises(tmp_path: Path) -> None:
    repo = GitRepo(tmp_path / "shadow")
    repo.init_if_needed()
    outside = tmp_path / "other.py"
    outside.write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(GitError, match="outside work_dir"):
        repo.commit_file(outside, "should not work")


def test_commit_file_relative_path(tmp_path: Path) -> None:
    """Relative paths are resolved against work_dir."""
    repo = GitRepo(tmp_path / "shadow")
    repo.init_if_needed()
    (repo.work_dir / "subdir").mkdir()
    (repo.work_dir / "subdir" / "f.py").write_text("y = 2\n", encoding="utf-8")
    sha = repo.commit_file(Path("subdir/f.py"), "rel")
    assert sha
