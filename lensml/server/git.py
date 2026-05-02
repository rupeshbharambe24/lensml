"""Git operations for the lensml session branch (plan §3.3).

Subprocess wrappers around the `git` CLI rather than dulwich/pygit2:
the operations we need (init / branch / add+commit / reset) are trivial
with the CLI, and the CLI is already on every dev machine.

Public API:
    GitRepo(work_dir).
        is_repo()          → True if work_dir is inside a git work tree
        init_if_needed()   → ensure a git repo exists; create one if not
        current_branch()   → e.g. "main"
        create_branch(n)   → checkout -b n
        commit_file(p, m)  → stage path and commit; returns SHA
        reset_hard(ref)    → git reset --hard ref (default HEAD~1)
        log_oneline(n)     → recent commits, one per line

All operations raise `GitError` on non-zero exit unless `check=False`.
The shadow-repo init pins user.email/user.name locally so the user's
global git config isn't required (works in clean CI environments).
"""

from __future__ import annotations

import logging
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


class GitError(RuntimeError):
    """Raised when a git command exits non-zero (and check=True)."""


@dataclass
class CommandResult:
    cmd: str
    stdout: str
    stderr: str
    returncode: int


def _run(cwd: Path, args: list[str], *, check: bool = True) -> CommandResult:
    cmd = ["git", *args]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as e:
        raise GitError(f"git CLI not found: {e}") from e

    result = CommandResult(
        cmd=" ".join(shlex.quote(c) for c in cmd),
        stdout=proc.stdout.strip(),
        stderr=proc.stderr.strip(),
        returncode=proc.returncode,
    )
    if check and result.returncode != 0:
        raise GitError(
            f"{result.cmd} failed (exit {result.returncode}): {result.stderr or result.stdout}"
        )
    return result


# Local config that pins identity (no dependence on the user's global config).
_IDENTITY = [
    "-c",
    "user.email=lensml@local",
    "-c",
    "user.name=lensml",
]


class GitRepo:
    """Thin wrapper around the git CLI for one work directory."""

    def __init__(self, work_dir: Path) -> None:
        self.work_dir = Path(work_dir).resolve()

    # ---- introspection ----

    def is_repo(self) -> bool:
        if not self.work_dir.exists():
            return False
        result = _run(self.work_dir, ["rev-parse", "--is-inside-work-tree"], check=False)
        return result.returncode == 0 and result.stdout == "true"

    def current_branch(self) -> str:
        return _run(self.work_dir, ["rev-parse", "--abbrev-ref", "HEAD"]).stdout

    def head_sha(self) -> str:
        return _run(self.work_dir, ["rev-parse", "HEAD"]).stdout

    def log_oneline(self, n: int = 10) -> list[str]:
        out = _run(self.work_dir, ["log", f"-{n}", "--oneline", "--no-color"], check=False)
        if out.returncode != 0:
            return []
        return out.stdout.splitlines()

    # ---- mutators ----

    def init_if_needed(self) -> bool:
        """Initialize a git repo at `work_dir` if it isn't already one.

        Returns True if a new repo was created. After init, makes a single
        seed commit so HEAD is valid (subsequent reset_hard(HEAD~1) won't
        underflow).
        """
        if self.is_repo():
            return False
        self.work_dir.mkdir(parents=True, exist_ok=True)
        _run(self.work_dir, ["init", "-b", "main"])
        gitignore = self.work_dir / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text("# lensml shadow repo\n", encoding="utf-8")
        _run(self.work_dir, ["add", ".gitignore"])
        _run(
            self.work_dir,
            [*_IDENTITY, "commit", "-m", "lensml: init shadow repo"],
        )
        return True

    def create_branch(self, name: str) -> None:
        """Create and switch to branch `name` from current HEAD."""
        _run(self.work_dir, ["checkout", "-b", name])

    def checkout(self, ref: str) -> None:
        _run(self.work_dir, ["checkout", ref])

    def commit_file(self, path: Path, message: str) -> str:
        """Stage `path` (absolute or relative to work_dir) and commit.

        Returns the new commit SHA.
        """
        path = Path(path)
        if path.is_absolute():
            try:
                rel = path.relative_to(self.work_dir)
            except ValueError as e:
                raise GitError(f"path {path} is outside work_dir {self.work_dir}") from e
        else:
            rel = path
        _run(self.work_dir, ["add", str(rel)])
        _run(self.work_dir, [*_IDENTITY, "commit", "-m", message])
        return self.head_sha()

    def reset_hard(self, ref: str = "HEAD~1") -> None:
        """git reset --hard ref."""
        _run(self.work_dir, ["reset", "--hard", ref])
