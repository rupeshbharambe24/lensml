"""Fixture loader for lensml-bench (plan §6.1).

Each fixture is a directory under ``lensml/bench/fixtures/<env>/<class>/<index>/``
containing four files:

    bug.py      — broken training script
    fix.diff    — canonical unified diff (apply to bug.py to produce a working
                  script). The lensml agent's job is to *propose* this diff;
                  the bench grader uses it as ground-truth.
    grade.py    — deterministic checks (subprocesses bug.py and the patched
                  version, asserts loss-converges + metric-thresholds). For v1
                  every grade.py is a thin shim over ``lensml.bench.grader``.
    meta.json   — {"pathology_class": str, "difficulty": "easy"|"medium"|"hard",
                    "host_env": "tiny"|"small"|"medium"|"large",
                    "max_steps": int, "loss_threshold": float,
                    "metric_threshold": float | null,
                    "metric_name": str | null}

This module knows how to discover, load, and validate fixtures. The
runner / grader / report modules consume the loaded ``Fixture`` objects.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

VALID_ENVS = ("tiny", "small", "medium", "large")
VALID_DIFFICULTIES = ("easy", "medium", "hard")
VALID_PATHOLOGY_CLASSES = (
    # 8 v1 pathologies (plan §4)
    "exploding_grad",
    "lr_too_high",
    "dead_relu",
    "missing_normalize",
    "dataloader_bottleneck",
    "model_eval_not_called",
    "bad_init",
    "vanishing_grad",
    # 4 stretch classes (plan §6.1) — diagnosis-only allowed for v1
    "label_leakage",
    "train_val_contamination",
    "mixed_precision_nan",
    "lr_schedule_mismatch",
)

REQUIRED_FILES = ("bug.py", "fix.diff", "grade.py", "meta.json")


class FixtureError(ValueError):
    """Raised when a fixture is missing files or has an invalid meta.json."""


@dataclass
class FixtureMeta:
    pathology_class: str
    difficulty: str
    host_env: str
    max_steps: int
    loss_threshold: float
    metric_threshold: float | None = None
    metric_name: str | None = None
    notes: str | None = None


@dataclass
class Fixture:
    """Loaded fixture handle."""

    path: Path
    meta: FixtureMeta
    bug_source: str
    fix_diff: str
    raw_meta: dict[str, Any] = field(default_factory=dict)

    @property
    def env(self) -> str:
        return self.meta.host_env

    @property
    def pathology(self) -> str:
        return self.meta.pathology_class

    @property
    def slug(self) -> str:
        """``<env>/<pathology>/<index>`` — stable, comparable, sortable."""
        return f"{self.meta.host_env}/{self.meta.pathology_class}/{self.path.name}"


def _validate_meta(raw: dict[str, Any], path: Path) -> FixtureMeta:
    required = {"pathology_class", "difficulty", "host_env", "max_steps", "loss_threshold"}
    missing = required - set(raw.keys())
    if missing:
        raise FixtureError(f"{path}: meta.json missing keys: {sorted(missing)}")

    pc = raw["pathology_class"]
    if pc not in VALID_PATHOLOGY_CLASSES:
        raise FixtureError(f"{path}: pathology_class={pc!r} not in {VALID_PATHOLOGY_CLASSES}")

    diff = raw["difficulty"]
    if diff not in VALID_DIFFICULTIES:
        raise FixtureError(f"{path}: difficulty={diff!r} not in {VALID_DIFFICULTIES}")

    env = raw["host_env"]
    if env not in VALID_ENVS:
        raise FixtureError(f"{path}: host_env={env!r} not in {VALID_ENVS}")

    max_steps = raw["max_steps"]
    if not isinstance(max_steps, int) or max_steps <= 0:
        raise FixtureError(f"{path}: max_steps must be a positive int")

    loss_threshold = raw["loss_threshold"]
    if not isinstance(loss_threshold, (int, float)):
        raise FixtureError(f"{path}: loss_threshold must be numeric")

    metric_threshold = raw.get("metric_threshold")
    if metric_threshold is not None and not isinstance(metric_threshold, (int, float)):
        raise FixtureError(f"{path}: metric_threshold must be numeric or null")

    return FixtureMeta(
        pathology_class=pc,
        difficulty=diff,
        host_env=env,
        max_steps=max_steps,
        loss_threshold=float(loss_threshold),
        metric_threshold=float(metric_threshold) if metric_threshold is not None else None,
        metric_name=raw.get("metric_name"),
        notes=raw.get("notes"),
    )


def load_fixture(path: Path) -> Fixture:
    """Load + validate a single fixture directory.

    Raises ``FixtureError`` if any required file is missing or
    ``meta.json`` is malformed.
    """
    if not path.is_dir():
        raise FixtureError(f"{path}: not a directory")

    for fname in REQUIRED_FILES:
        if not (path / fname).is_file():
            raise FixtureError(f"{path}: missing {fname}")

    raw_meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    if not isinstance(raw_meta, dict):
        raise FixtureError(f"{path}: meta.json must be a JSON object")
    meta = _validate_meta(raw_meta, path)

    # The path layout encodes env + pathology too — verify consistency.
    parts = path.parts
    if len(parts) >= 3:
        env_from_path = parts[-3]
        pathology_from_path = parts[-2]
        if env_from_path in VALID_ENVS and env_from_path != meta.host_env:
            raise FixtureError(
                f"{path}: meta host_env={meta.host_env} but path implies {env_from_path}"
            )
        if (
            pathology_from_path in VALID_PATHOLOGY_CLASSES
            and pathology_from_path != meta.pathology_class
        ):
            raise FixtureError(
                f"{path}: meta pathology_class={meta.pathology_class} "
                f"but path implies {pathology_from_path}"
            )

    bug_source = (path / "bug.py").read_text(encoding="utf-8")
    fix_diff = (path / "fix.diff").read_text(encoding="utf-8")

    return Fixture(
        path=path,
        meta=meta,
        bug_source=bug_source,
        fix_diff=fix_diff,
        raw_meta=raw_meta,
    )


def discover_fixtures(root: Path | None = None) -> list[Fixture]:
    """Walk the fixtures/ tree and return every well-formed fixture.

    The default root is the in-package ``lensml/bench/fixtures/`` directory.
    Pass a different ``root`` (e.g. a tmp_path) for tests.

    Skips dot-directories and the top-level README.md.
    """
    root = root if root is not None else _default_fixtures_root()
    fixtures: list[Fixture] = []
    if not root.exists():
        return fixtures
    for env_dir in sorted(root.iterdir()):
        if not env_dir.is_dir() or env_dir.name.startswith("."):
            continue
        if env_dir.name not in VALID_ENVS:
            continue
        for pathology_dir in sorted(env_dir.iterdir()):
            if not pathology_dir.is_dir():
                continue
            for index_dir in sorted(pathology_dir.iterdir()):
                if not index_dir.is_dir():
                    continue
                if not (index_dir / "meta.json").is_file():
                    continue
                fixtures.append(load_fixture(index_dir))
    return fixtures


def _default_fixtures_root() -> Path:
    return Path(__file__).resolve().parent / "fixtures"


__all__ = [
    "REQUIRED_FILES",
    "VALID_DIFFICULTIES",
    "VALID_ENVS",
    "VALID_PATHOLOGY_CLASSES",
    "Fixture",
    "FixtureError",
    "FixtureMeta",
    "discover_fixtures",
    "load_fixture",
]
