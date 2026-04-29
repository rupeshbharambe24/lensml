"""Integration tests for the dev `_test_publish` endpoint."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from lensml.server.app import create_app


@pytest.fixture
def client() -> TestClient:
    app = create_app(db_path=":memory:")
    with TestClient(app) as c:
        yield c


def test_test_publish_creates_run(client: TestClient) -> None:
    """First publish auto-creates the run row."""
    r = client.post(
        "/api/runs/demo/_test_publish",
        json={"step": 1, "loss": 0.5},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "published"
    assert body["run_id"] == "demo"
    assert body["kind"] == "metric"

    runs = client.get("/api/runs").json()
    assert any(run["id"] == "demo" for run in runs)


def test_test_publish_appends_event(client: TestClient) -> None:
    client.post("/api/runs/demo/_test_publish", json={"step": 0, "loss": 1.0})
    client.post("/api/runs/demo/_test_publish", json={"step": 1, "loss": 0.9})
    events = client.get("/api/runs/demo/events").json()
    assert len(events) == 2
    assert events[0]["payload"]["step"] == 0
    assert events[1]["payload"]["loss"] == 0.9


def test_test_publish_uses_path_run_id_when_missing(client: TestClient) -> None:
    """Body without explicit run_id inherits the path's run_id."""
    client.post("/api/runs/auto/_test_publish", json={"step": 0})
    events = client.get("/api/runs/auto/events").json()
    assert events[0]["payload"]["run_id"] == "auto"


def test_test_publish_custom_kind(client: TestClient) -> None:
    """Caller can set kind explicitly."""
    r = client.post(
        "/api/runs/demo/_test_publish",
        json={"kind": "anomaly", "name": "exploding_grad"},
    )
    assert r.json()["kind"] == "anomaly"
    events = client.get("/api/runs/demo/events", params={"kind": "anomaly"}).json()
    assert len(events) == 1
