"""Unit tests for lensml.bench.fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lensml.bench.fixtures import (
    REQUIRED_FILES,
    VALID_DIFFICULTIES,
    VALID_ENVS,
    VALID_PATHOLOGY_CLASSES,
    Fixture,
    FixtureError,
    discover_fixtures,
    load_fixture,
)


def _write_fixture(
    base: Path,
    *,
    env: str = "tiny",
    pathology: str = "exploding_grad",
    index: str = "0",
    bug: str = "print('bug')\n",
    diff: str = "",
    grade: str = "print('grade')\n",
    meta: dict | None = None,
) -> Path:
    p = base / env / pathology / index
    p.mkdir(parents=True, exist_ok=True)
    (p / "bug.py").write_text(bug, encoding="utf-8")
    (p / "fix.diff").write_text(diff, encoding="utf-8")
    (p / "grade.py").write_text(grade, encoding="utf-8")
    full_meta = {
        "pathology_class": pathology,
        "difficulty": "easy",
        "host_env": env,
        "max_steps": 100,
        "loss_threshold": 0.5,
    }
    if meta:
        full_meta.update(meta)
    (p / "meta.json").write_text(json.dumps(full_meta), encoding="utf-8")
    return p


def test_required_files_constant() -> None:
    assert REQUIRED_FILES == ("bug.py", "fix.diff", "grade.py", "meta.json")


def test_valid_constants_match_plan() -> None:
    # 12 pathology classes per plan §6.1 (8 v1 + 4 stretch).
    assert len(VALID_PATHOLOGY_CLASSES) == 12
    assert "exploding_grad" in VALID_PATHOLOGY_CLASSES
    assert "lr_schedule_mismatch" in VALID_PATHOLOGY_CLASSES
    assert set(VALID_ENVS) == {"tiny", "small", "medium", "large"}
    assert set(VALID_DIFFICULTIES) == {"easy", "medium", "hard"}


def test_load_fixture_round_trip(tmp_path: Path) -> None:
    p = _write_fixture(tmp_path)
    fix = load_fixture(p)
    assert isinstance(fix, Fixture)
    assert fix.meta.pathology_class == "exploding_grad"
    assert fix.meta.host_env == "tiny"
    assert fix.bug_source == "print('bug')\n"
    assert fix.slug == "tiny/exploding_grad/0"
    assert fix.env == "tiny"
    assert fix.pathology == "exploding_grad"


def test_load_fixture_missing_file_raises(tmp_path: Path) -> None:
    p = _write_fixture(tmp_path)
    (p / "fix.diff").unlink()
    with pytest.raises(FixtureError) as exc:
        load_fixture(p)
    assert "fix.diff" in str(exc.value)


def test_load_fixture_unknown_pathology_raises(tmp_path: Path) -> None:
    p = _write_fixture(tmp_path, meta={"pathology_class": "nope"})
    with pytest.raises(FixtureError):
        load_fixture(p)


def test_load_fixture_unknown_env_raises(tmp_path: Path) -> None:
    p = _write_fixture(tmp_path, meta={"host_env": "huge"})
    with pytest.raises(FixtureError):
        load_fixture(p)


def test_load_fixture_unknown_difficulty_raises(tmp_path: Path) -> None:
    p = _write_fixture(tmp_path, meta={"difficulty": "trivial"})
    with pytest.raises(FixtureError):
        load_fixture(p)


def test_load_fixture_negative_max_steps_raises(tmp_path: Path) -> None:
    p = _write_fixture(tmp_path, meta={"max_steps": 0})
    with pytest.raises(FixtureError):
        load_fixture(p)


def test_load_fixture_path_meta_mismatch_raises(tmp_path: Path) -> None:
    """A fixture under tiny/exploding_grad/0/ with meta.host_env='small'
    is rejected — the filesystem layout must match the metadata."""
    p = _write_fixture(tmp_path, env="tiny", meta={"host_env": "small"})
    with pytest.raises(FixtureError):
        load_fixture(p)


def test_discover_fixtures_picks_up_well_formed(tmp_path: Path) -> None:
    _write_fixture(tmp_path, pathology="exploding_grad", index="0")
    _write_fixture(tmp_path, pathology="dead_relu", index="0")
    _write_fixture(tmp_path, env="small", pathology="bad_init", index="0")
    found = discover_fixtures(tmp_path)
    assert {f.slug for f in found} == {
        "tiny/exploding_grad/0",
        "tiny/dead_relu/0",
        "small/bad_init/0",
    }


def test_discover_fixtures_skips_non_env_dirs(tmp_path: Path) -> None:
    (tmp_path / ".cache").mkdir()
    (tmp_path / "scratch").mkdir()
    _write_fixture(tmp_path, pathology="exploding_grad", index="0")
    found = discover_fixtures(tmp_path)
    assert len(found) == 1


def test_discover_fixtures_skips_dirs_without_meta(tmp_path: Path) -> None:
    p = tmp_path / "tiny" / "exploding_grad" / "broken"
    p.mkdir(parents=True)
    (p / "bug.py").write_text("x", encoding="utf-8")
    found = discover_fixtures(tmp_path)
    assert found == []


def test_discover_fixtures_default_root_finds_packaged_fixtures() -> None:
    """The packaged tiny fixtures must all load. Step 4.1 shipped 4;
    step 4.2 added 4 more so all 8 v1 detector classes (plus one
    stretch class) have working fixtures."""
    found = discover_fixtures()
    slugs = {f.slug for f in found}
    expected = {
        "tiny/exploding_grad/0",
        "tiny/dead_relu/0",
        "tiny/missing_normalize/0",
        "tiny/lr_too_high/0",
        "tiny/bad_init/0",
        "tiny/vanishing_grad/0",
        "tiny/model_eval_not_called/0",
        "tiny/lr_schedule_mismatch/0",
    }
    assert expected <= slugs, f"missing: {expected - slugs}"
    for f in found:
        assert f.meta.host_env in VALID_ENVS
        assert f.meta.pathology_class in VALID_PATHOLOGY_CLASSES


def test_all_v1_detector_classes_have_tiny_fixture() -> None:
    """Every detector class lensml ships with v1 must have at least one
    tiny fixture so the bench has full v1 coverage."""
    v1_detectors = {
        "exploding_grad",
        "lr_too_high",
        "dead_relu",
        "missing_normalize",
        "model_eval_not_called",
        "bad_init",
        "vanishing_grad",
    }
    # dataloader_bottleneck is structural / I/O-bound — lives in small/medium
    # envs where there's real disk contention. Documented as out-of-scope
    # for the tiny bench in lensml/bench/fixtures/README.md.

    found = discover_fixtures()
    tiny_classes = {f.meta.pathology_class for f in found if f.meta.host_env == "tiny"}
    missing = v1_detectors - tiny_classes
    assert not missing, f"v1 detector classes without a tiny fixture: {sorted(missing)}"
