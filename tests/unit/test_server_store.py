"""Unit tests for lensml.server.store."""

from __future__ import annotations

import time

import pytest

from lensml.server.store import EventStore


@pytest.fixture
def store() -> EventStore:
    s = EventStore(":memory:")
    yield s
    s.close()


# ---- runs ----


def test_begin_run_inserts(store: EventStore) -> None:
    store.begin_run("run-1", script_path="train.py", git_branch="lensml/abc")
    run = store.get_run("run-1")
    assert run is not None
    assert run["id"] == "run-1"
    assert run["script_path"] == "train.py"
    assert run["git_branch"] == "lensml/abc"
    assert run["status"] == "running"
    assert run["ended_at"] is None
    assert run["started_at"] > 0


def test_begin_run_idempotent(store: EventStore) -> None:
    store.begin_run("run-1", script_path="a.py")
    store.begin_run("run-1", script_path="b.py")
    run = store.get_run("run-1")
    assert run is not None
    assert run["script_path"] == "a.py"


def test_end_run_sets_status(store: EventStore) -> None:
    store.begin_run("run-1")
    store.end_run("run-1", status="completed")
    run = store.get_run("run-1")
    assert run is not None
    assert run["status"] == "completed"
    assert run["ended_at"] is not None


def test_get_run_missing(store: EventStore) -> None:
    assert store.get_run("nope") is None


def test_list_runs_ordered_by_recent_first(store: EventStore) -> None:
    store.begin_run("a")
    time.sleep(0.001)
    store.begin_run("b")
    time.sleep(0.001)
    store.begin_run("c")
    runs = store.list_runs()
    assert [r["id"] for r in runs] == ["c", "b", "a"]


# ---- events ----


def test_append_event_returns_id(store: EventStore) -> None:
    store.begin_run("r")
    eid1 = store.append_event("r", "metric", {"loss": 1.0})
    eid2 = store.append_event("r", "metric", {"loss": 0.9})
    assert eid2 > eid1


def test_get_events_orders_by_id(store: EventStore) -> None:
    store.begin_run("r")
    for i in range(5):
        store.append_event("r", "metric", {"step": i})
    events = store.get_events("r")
    assert len(events) == 5
    assert [e["payload"]["step"] for e in events] == [0, 1, 2, 3, 4]


def test_get_events_since_id(store: EventStore) -> None:
    store.begin_run("r")
    ids = [store.append_event("r", "metric", {"step": i}) for i in range(5)]
    events = store.get_events("r", since_id=ids[2])
    assert [e["payload"]["step"] for e in events] == [3, 4]


def test_get_events_filter_by_kind(store: EventStore) -> None:
    store.begin_run("r")
    store.append_event("r", "metric", {"loss": 1.0})
    store.append_event("r", "anomaly", {"kind": "exploding_grad"})
    store.append_event("r", "metric", {"loss": 0.5})
    only = store.get_events("r", kind="anomaly")
    assert len(only) == 1
    assert only[0]["kind"] == "anomaly"


def test_get_events_limit(store: EventStore) -> None:
    store.begin_run("r")
    for i in range(20):
        store.append_event("r", "metric", {"step": i})
    events = store.get_events("r", limit=5)
    assert len(events) == 5


def test_event_count(store: EventStore) -> None:
    store.begin_run("r")
    for i in range(7):
        store.append_event("r", "metric", {"step": i})
    assert store.event_count("r") == 7


def test_payload_round_trips_json(store: EventStore) -> None:
    """Nested dicts and arrays survive JSON encoding."""
    store.begin_run("r")
    payload = {"layers": [{"name": "fc1", "stats": {"mean": 0.1}}], "step": 42}
    store.append_event("r", "metric", payload)
    events = store.get_events("r")
    assert events[0]["payload"] == payload


# ---- decisions ----


def test_record_and_resolve_decision(store: EventStore) -> None:
    store.begin_run("r")
    did = store.record_decision("r", "AddGradientClipping", diff_text="+ clip")
    decisions = store.list_decisions("r")
    assert len(decisions) == 1
    assert decisions[0]["fix_template"] == "AddGradientClipping"
    assert decisions[0]["approved"] is None
    store.resolve_decision(did, approved=True)
    decisions = store.list_decisions("r")
    assert decisions[0]["approved"] == 1
    assert decisions[0]["applied_at"] is not None


def test_resolve_decision_rejected(store: EventStore) -> None:
    store.begin_run("r")
    did = store.record_decision("r", "LowerLearningRate")
    store.resolve_decision(did, approved=False)
    decisions = store.list_decisions("r")
    assert decisions[0]["approved"] == 0
    assert decisions[0]["applied_at"] is None
