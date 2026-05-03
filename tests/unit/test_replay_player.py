"""Unit tests for lensml.replay.player."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

from lensml.replay.player import (
    DEFAULT_STEP_DELAY,
    MAX_SLEEP_SECONDS,
    Player,
    _pacing_iter,
    read_jsonl,
    write_jsonl,
)

# ---- read_jsonl / write_jsonl ----


def test_write_then_read_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "session.jsonl"
    write_jsonl(p, {"kind": "metric", "step": 1, "loss": 0.5})
    write_jsonl(p, {"kind": "anomaly", "step": 1, "detector": "exploding_grad"})
    events = read_jsonl(p)
    assert len(events) == 2
    assert events[0]["kind"] == "metric"
    assert events[1]["detector"] == "exploding_grad"


def test_read_jsonl_skips_blank_and_malformed(tmp_path: Path) -> None:
    p = tmp_path / "messy.jsonl"
    p.write_text('\n{"kind": "metric"}\nnot-json\n\n{"kind": "anomaly"}\n', encoding="utf-8")
    events = read_jsonl(p)
    assert len(events) == 2
    assert events[0]["kind"] == "metric"
    assert events[1]["kind"] == "anomaly"


def test_read_jsonl_skips_top_level_arrays(tmp_path: Path) -> None:
    p = tmp_path / "list.jsonl"
    p.write_text('[1, 2]\n{"kind": "metric"}\n', encoding="utf-8")
    events = read_jsonl(p)
    assert len(events) == 1
    assert events[0]["kind"] == "metric"


def test_write_jsonl_creates_parent_dirs(tmp_path: Path) -> None:
    p = tmp_path / "nested" / "deep" / "session.jsonl"
    write_jsonl(p, {"kind": "metric", "step": 0})
    assert p.exists()


# ---- _pacing_iter ----


def test_pacing_first_event_has_zero_sleep() -> None:
    pairs = list(
        _pacing_iter(
            [{"kind": "metric", "ts": 100.0}, {"kind": "metric", "ts": 100.5}],
            fixed_delay=1.0,
            speed=1.0,
        )
    )
    assert len(pairs) == 2
    assert pairs[0][1] == 0.0
    assert pairs[1][1] == pytest.approx(0.5)


def test_pacing_uses_fixed_when_no_ts() -> None:
    pairs = list(
        _pacing_iter(
            [{"kind": "metric"}, {"kind": "metric"}, {"kind": "metric"}],
            fixed_delay=0.1,
            speed=1.0,
        )
    )
    assert pairs[0][1] == 0.0
    assert pairs[1][1] == pytest.approx(0.1)
    assert pairs[2][1] == pytest.approx(0.1)


def test_pacing_speed_divides_sleeps() -> None:
    pairs = list(
        _pacing_iter(
            [{"kind": "metric", "ts": 0.0}, {"kind": "metric", "ts": 2.0}],
            fixed_delay=1.0,
            speed=4.0,
        )
    )
    assert pairs[1][1] == pytest.approx(0.5)


def test_pacing_clamps_to_max_sleep() -> None:
    pairs = list(
        _pacing_iter(
            [{"kind": "metric", "ts": 0.0}, {"kind": "metric", "ts": 1000.0}],
            fixed_delay=0.0,
            speed=1.0,
        )
    )
    assert pairs[1][1] == MAX_SLEEP_SECONDS


def test_pacing_reverse_ts_falls_back_to_fixed() -> None:
    pairs = list(
        _pacing_iter(
            [{"kind": "metric", "ts": 100.0}, {"kind": "metric", "ts": 50.0}],
            fixed_delay=0.2,
            speed=1.0,
        )
    )
    assert pairs[1][1] == pytest.approx(0.2)


# ---- Player ----


def test_player_constructor_validates_speed() -> None:
    with pytest.raises(ValueError):
        Player(speed=0)
    with pytest.raises(ValueError):
        Player(speed=-1)


def test_player_constructor_validates_delay() -> None:
    with pytest.raises(ValueError):
        Player(fixed_delay=-0.1)


def test_play_dispatches_in_order() -> None:
    seen: list[dict] = []
    Player(fixed_delay=0.0, speed=1.0).play(
        [
            {"kind": "metric", "step": 0},
            {"kind": "metric", "step": 1},
            {"kind": "anomaly", "step": 1},
        ],
        dispatch=seen.append,
    )
    assert [e["kind"] for e in seen] == ["metric", "metric", "anomaly"]
    assert [e["step"] for e in seen] == [0, 1, 1]


def test_play_overrides_run_id() -> None:
    seen: list[dict] = []
    Player(fixed_delay=0.0, run_id="override").play(
        [{"kind": "metric", "run_id": "original", "step": 0}], dispatch=seen.append
    )
    assert seen[0]["run_id"] == "override"


def test_play_does_not_mutate_caller_events() -> None:
    src = [{"kind": "metric", "run_id": "original", "step": 0}]
    Player(fixed_delay=0.0, run_id="override").play(src, dispatch=lambda e: None)
    assert src[0]["run_id"] == "original"


def test_play_continues_through_dispatch_errors() -> None:
    seen: list[dict] = []

    def bad_dispatch(ev: dict) -> None:
        if ev["step"] == 1:
            raise RuntimeError("boom")
        seen.append(ev)

    stats = Player(fixed_delay=0.0).play(
        [
            {"kind": "metric", "step": 0},
            {"kind": "metric", "step": 1},
            {"kind": "metric", "step": 2},
        ],
        dispatch=bad_dispatch,
    )
    assert [e["step"] for e in seen] == [0, 2]
    assert stats.dispatched == 2
    assert stats.skipped == 1


def test_play_records_stats_kinds() -> None:
    stats = Player(fixed_delay=0.0).play(
        [
            {"kind": "metric"},
            {"kind": "metric"},
            {"kind": "anomaly"},
        ],
        dispatch=lambda e: None,
    )
    assert stats.dispatched == 3
    assert stats.kinds == {"metric": 2, "anomaly": 1}


def test_play_pacing_uses_real_sleep() -> None:
    """Playing 3 events with fixed_delay=0.05 should take ≈ 0.1 s
    (sleeps between events 1→2 and 2→3, none before event 1)."""
    t0 = time.time()
    Player(fixed_delay=0.05, speed=1.0).play(
        [{"kind": "metric"}, {"kind": "metric"}, {"kind": "metric"}],
        dispatch=lambda e: None,
    )
    elapsed = time.time() - t0
    assert 0.08 < elapsed < 0.5  # generous bounds for CI variance


@pytest.mark.asyncio
async def test_play_async_supports_async_dispatch() -> None:
    seen: list[dict] = []

    async def dispatch(ev: dict) -> None:
        await asyncio.sleep(0)
        seen.append(ev)

    stats = await Player(fixed_delay=0.0).play_async(
        [{"kind": "metric", "step": 0}, {"kind": "metric", "step": 1}],
        dispatch=dispatch,
    )
    assert [e["step"] for e in seen] == [0, 1]
    assert stats.dispatched == 2


@pytest.mark.asyncio
async def test_play_async_supports_sync_dispatch() -> None:
    seen: list[dict] = []
    await Player(fixed_delay=0.0).play_async([{"kind": "metric", "step": 0}], dispatch=seen.append)
    assert len(seen) == 1


def test_default_step_delay_matches_plan() -> None:
    """Plan §3.1 envelope is 10–50 Hz. 50 ms ≈ 20 Hz is in the middle."""
    assert DEFAULT_STEP_DELAY == 0.05


def test_jsonl_lines_are_valid_json(tmp_path: Path) -> None:
    """A line written by write_jsonl is parseable on its own."""
    p = tmp_path / "x.jsonl"
    write_jsonl(p, {"a": 1})
    write_jsonl(p, {"b": 2})
    raw = p.read_text(encoding="utf-8").splitlines()
    assert json.loads(raw[0]) == {"a": 1}
    assert json.loads(raw[1]) == {"b": 2}
