"""FastAPI application factory.

Plan §3.1 / §8 Day 3:
    - SSE endpoints stream live training metrics from the ZMQ SUB bridge.
    - SQLite event log (lensml.server.store) is the persistent record.
    - REST endpoints (under /api/runs) expose run metadata + decision queue.

Default bind: 127.0.0.1 (do NOT expose externally; see SECURITY.md).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import zmq
import zmq.asyncio
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from lensml.detectors.exploding_grad import ExplodingGradDetector
from lensml.detectors.runner import DetectorRunner
from lensml.server.sse import EventBroker
from lensml.server.store import EventStore

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from lensml.detectors.base import Detector

log = logging.getLogger(__name__)


async def _zmq_bridge(
    endpoint: str,
    broker: EventBroker,
    store: EventStore,
) -> None:
    """Bridge ZMQ SUB → broker (live SSE) and store (persistence).

    Runs as a background asyncio task for the lifetime of the app. Each
    event coming off the SUB socket is appended to the SQLite event log
    (run_id and kind extracted from the payload, defaulting to
    "default"/"metric") and broadcast to all SSE subscribers.
    """
    ctx = zmq.asyncio.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect(endpoint)
    sock.setsockopt(zmq.SUBSCRIBE, b"")
    log.info("zmq bridge connected: %s", endpoint)
    try:
        while True:
            event: dict[str, Any] = await sock.recv_json()
            run_id = str(event.get("run_id", "default"))
            kind = str(event.get("kind", "metric"))
            try:
                store.append_event(run_id, kind, event)
            except Exception:
                log.exception("event store append failed")
            broker.publish(event)
    except asyncio.CancelledError:
        log.info("zmq bridge cancelled")
    finally:
        sock.close(linger=0)
        ctx.term()


DEFAULT_CORS_ORIGINS: tuple[str, ...] = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
)


def _default_detectors() -> list[Detector]:
    """v1 detector roster (plan §4 row 1 — others land in steps 3.3-3.4)."""
    return [ExplodingGradDetector()]


def create_app(
    *,
    db_path: str = ":memory:",
    zmq_endpoint: str | None = None,
    cors_origins: tuple[str, ...] | list[str] = DEFAULT_CORS_ORIGINS,
    detectors: list[Detector] | None = None,
) -> FastAPI:
    """Application factory.

    Args:
        db_path: SQLite file (or ``:memory:`` for tests).
        zmq_endpoint: if set, spawn a background ZMQ SUB → broker bridge.
            None disables the bridge — useful for tests that publish
            directly to the broker.
        cors_origins: list of allowed origins for browser CORS. Default
            allows the local Next.js dev server. SSE specifically NEEDS
            CORS rather than going through the Next.js dev proxy because
            the dev proxy buffers streaming responses.
        detectors: which detector instances to run against the broker
            stream. Defaults to the v1 roster (plan §4); pass ``[]`` to
            disable detection (useful in tests that exercise transport
            only).
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        store = EventStore(db_path)
        broker = EventBroker()
        bridge_task: asyncio.Task[None] | None = None

        if zmq_endpoint is not None:
            bridge_task = asyncio.create_task(_zmq_bridge(zmq_endpoint, broker, store))

        runner = DetectorRunner(
            broker, store, detectors if detectors is not None else _default_detectors()
        )
        await runner.start()

        app.state.store = store
        app.state.broker = broker
        app.state.detector_runner = runner
        try:
            yield
        finally:
            await runner.stop()
            if bridge_task is not None:
                bridge_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await bridge_task
            store.close()

    app = FastAPI(title="lensml", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/healthz")
    async def healthz() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/api/runs")
    async def list_runs(request: Request) -> list[dict[str, Any]]:
        store: EventStore = request.app.state.store
        return store.list_runs()

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str, request: Request) -> dict[str, Any]:
        store: EventStore = request.app.state.store
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"run {run_id} not found")
        return run

    @app.get("/api/runs/{run_id}/events")
    async def get_events(
        run_id: str,
        request: Request,
        since_id: int = 0,
        limit: int = 1000,
        kind: str | None = None,
    ) -> list[dict[str, Any]]:
        store: EventStore = request.app.state.store
        return store.get_events(run_id, since_id=since_id, limit=limit, kind=kind)

    @app.post("/api/runs/{run_id}/_test_publish")
    async def test_publish(run_id: str, request: Request) -> dict[str, Any]:
        """Dev-only helper: inject a test event into the broker + store.

        Lets you smoke-test the dashboard SSE pipeline without running a real
        trainer. Body is any JSON dict; ``run_id`` and ``kind`` are filled in
        from the path / default if absent.

            curl.exe -X POST http://127.0.0.1:8000/api/runs/demo-run/_test_publish ^
              -H "Content-Type: application/json" ^
              -d "{\"step\": 1, \"loss\": 0.5}"
        """
        payload: dict[str, Any] = await request.json()
        payload.setdefault("run_id", run_id)
        kind = str(payload.setdefault("kind", "metric"))
        store: EventStore = request.app.state.store
        broker: EventBroker = request.app.state.broker
        store.begin_run(run_id)  # idempotent — safe if already exists
        store.append_event(run_id, kind, payload)
        broker.publish(payload)
        return {
            "status": "published",
            "run_id": run_id,
            "kind": kind,
            "subscribers": broker.subscriber_count,
        }

    @app.get("/api/runs/{run_id}/metrics")
    async def metrics_stream(run_id: str, request: Request) -> EventSourceResponse:
        broker: EventBroker = request.app.state.broker
        q = broker.subscribe()

        async def event_gen() -> AsyncIterator[dict[str, str]]:
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        event = await asyncio.wait_for(q.get(), timeout=15.0)
                    except TimeoutError:
                        # Heartbeat to keep proxies and EventSource happy.
                        yield {"event": "ping", "data": "{}"}
                        continue
                    if event.get("run_id", "default") == run_id:
                        yield {"data": json.dumps(event, default=str)}
            finally:
                broker.unsubscribe(q)

        return EventSourceResponse(event_gen())

    return app
