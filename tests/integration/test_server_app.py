"""Integration tests for lensml.server.app.

The streaming SSE endpoint is intentionally NOT exercised through TestClient
in this suite. Starlette's TestClient uses a synchronous httpx adapter that
doesn't reliably break out of an open SSE stream — tests deadlock waiting
for the server to close. The broker mechanics are covered by direct unit
tests in `test_server_sse.py`; the SSE wire-format goes through
``sse_starlette.EventSourceResponse`` which has its own test coverage.

End-to-end SSE (browser ↔ server) is verified manually during build via:

    curl -N http://127.0.0.1:8000/api/runs/{id}/metrics

per plan §8 Day 3 acceptance criterion.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from lensml.server.app import create_app


@pytest.fixture
def client() -> TestClient:
    app = create_app(db_path=":memory:")
    with TestClient(app) as c:
        yield c


def test_healthz(client: TestClient) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_list_runs_empty(client: TestClient) -> None:
    r = client.get("/api/runs")
    assert r.status_code == 200
    assert r.json() == []


def test_create_and_list_runs(client: TestClient) -> None:
    """Store reachable via app.state — write directly, then GET."""
    client.app.state.store.begin_run("run-1", script_path="train.py", git_branch="main")
    r = client.get("/api/runs")
    assert r.status_code == 200
    runs = r.json()
    assert len(runs) == 1
    assert runs[0]["id"] == "run-1"


def test_get_run_404(client: TestClient) -> None:
    r = client.get("/api/runs/nope")
    assert r.status_code == 404


def test_get_run_ok(client: TestClient) -> None:
    client.app.state.store.begin_run("run-1")
    r = client.get("/api/runs/run-1")
    assert r.status_code == 200
    assert r.json()["id"] == "run-1"


def test_get_events(client: TestClient) -> None:
    store = client.app.state.store
    store.begin_run("run-1")
    for i in range(3):
        store.append_event("run-1", "metric", {"step": i, "loss": 1.0 - i * 0.1})
    r = client.get("/api/runs/run-1/events")
    assert r.status_code == 200
    events = r.json()
    assert len(events) == 3
    assert events[0]["payload"]["step"] == 0


def test_get_events_filter_kind(client: TestClient) -> None:
    store = client.app.state.store
    store.begin_run("run-1")
    store.append_event("run-1", "metric", {"loss": 0.1})
    store.append_event("run-1", "anomaly", {"kind": "exploding_grad"})
    r = client.get("/api/runs/run-1/events", params={"kind": "anomaly"})
    assert r.status_code == 200
    events = r.json()
    assert len(events) == 1
    assert events[0]["kind"] == "anomaly"


def test_get_events_since_id(client: TestClient) -> None:
    store = client.app.state.store
    store.begin_run("run-1")
    ids = [store.append_event("run-1", "metric", {"step": i}) for i in range(5)]
    r = client.get("/api/runs/run-1/events", params={"since_id": ids[2]})
    events = r.json()
    assert [e["payload"]["step"] for e in events] == [3, 4]


def test_app_state_has_broker(client: TestClient) -> None:
    """Broker is wired up at startup so endpoints can fan-out events."""
    from lensml.server.sse import EventBroker

    assert isinstance(client.app.state.broker, EventBroker)
