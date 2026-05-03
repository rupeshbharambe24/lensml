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
from pathlib import Path
from typing import TYPE_CHECKING, Any

import zmq
import zmq.asyncio
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from lensml.detectors.exploding_grad import ExplodingGradDetector
from lensml.detectors.runner import DetectorRunner
from lensml.server.decisions import apply_decision, compute_diff_preview
from lensml.server.git import GitError, GitRepo
from lensml.server.privacy import PrivacyConfig, canned_diagnosis
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
    script_path: str | None = None,
    repo_path: str | None = None,
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
        app.state.script_path = script_path
        app.state.repo_path = repo_path
        app.state.privacy = PrivacyConfig()
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

    @app.get("/api/privacy")
    async def get_privacy(request: Request) -> dict[str, Any]:
        """Read the current privacy-mode config (plan §3.7)."""
        cfg: PrivacyConfig = request.app.state.privacy
        return cfg.to_dict()

    @app.post("/api/privacy")
    async def set_privacy(request: Request) -> dict[str, Any]:
        """Toggle privacy mode (plan §3.7).

        Body: ``{"enabled": bool}``. The toggle is informational for v1
        — see ``lensml/server/privacy.py`` and plan §5.6 for the full
        Plan A/B/C tradeoff.
        """
        cfg: PrivacyConfig = request.app.state.privacy
        body = await request.json()
        if not isinstance(body, dict) or "enabled" not in body:
            raise HTTPException(status_code=400, detail="body must be {'enabled': bool}")
        enabled = body["enabled"]
        if not isinstance(enabled, bool):
            raise HTTPException(status_code=400, detail="'enabled' must be a boolean")
        cfg.set_enabled(enabled)
        return cfg.to_dict()

    @app.post("/api/privacy/canned")
    async def privacy_canned(request: Request) -> dict[str, Any]:
        """Return a canned local-mode diagnosis for a given anomaly
        (plan §3.7 — Plan C: one canned interaction works).

        Body: ``{"anomaly": {...}}`` — the same shape as a detector's
        ``Anomaly.to_event()`` output. No network call is made; the
        response is computed from a static mapping in
        ``lensml.server.privacy``. This is the privacy-mode guarantee.
        """
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="body must be JSON object")
        anomaly = body.get("anomaly")
        if not isinstance(anomaly, dict):
            raise HTTPException(status_code=400, detail="body must contain {'anomaly': {...}}")
        return canned_diagnosis(anomaly)

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

    # ---- decision approval flow (plan §3.3 / §8 D12) ----

    def _resolve_script_path(request: Request, override: str | None) -> Path | None:
        """Choose the script path: request body > app config > None."""
        from pathlib import Path as _Path

        if override:
            return _Path(override)
        configured = request.app.state.script_path
        return _Path(configured) if configured else None

    def _resolve_repo(request: Request) -> GitRepo | None:
        from pathlib import Path as _Path

        repo_path = request.app.state.repo_path
        if not repo_path:
            return None
        return GitRepo(_Path(repo_path))

    @app.get("/api/runs/{run_id}/decisions")
    async def list_decisions(run_id: str, request: Request) -> list[dict[str, Any]]:
        store: EventStore = request.app.state.store
        return store.list_decisions(run_id)

    @app.get("/api/runs/{run_id}/decisions/{decision_id}")
    async def get_decision(
        run_id: str,
        decision_id: int,
        request: Request,
        script_path: str | None = None,
    ) -> dict[str, Any]:
        store: EventStore = request.app.state.store
        decision = store.get_decision(decision_id)
        if decision is None or decision.get("run_id") != run_id:
            raise HTTPException(status_code=404, detail=f"decision {decision_id} not found")

        path = _resolve_script_path(request, script_path)
        out: dict[str, Any] = dict(decision)
        if path is None:
            out["preview"] = {
                "applicable": False,
                "reason": "no script_path configured (set --script-path on the server or pass ?script_path=)",
            }
            return out

        preview = compute_diff_preview(decision, path)
        out["preview"] = {
            "applicable": preview.applicable,
            "reason": preview.reason,
            "template": preview.template,
            "rationale": preview.rationale,
            "params": preview.params,
            "old_source": preview.old_source,
            "new_source": preview.new_source,
            "diff": preview.diff,
        }
        return out

    @app.post("/api/runs/{run_id}/decisions/{decision_id}/approve")
    async def approve_decision(
        run_id: str,
        decision_id: int,
        request: Request,
    ) -> dict[str, Any]:
        body = {}
        try:
            body = await request.json()
        except Exception:
            body = {}
        script_override = body.get("script_path") if isinstance(body, dict) else None

        store: EventStore = request.app.state.store
        broker: EventBroker = request.app.state.broker
        decision = store.get_decision(decision_id)
        if decision is None or decision.get("run_id") != run_id:
            raise HTTPException(status_code=404, detail=f"decision {decision_id} not found")
        if decision.get("approved") is not None:
            raise HTTPException(
                status_code=409,
                detail=f"decision {decision_id} already resolved",
            )

        path = _resolve_script_path(request, script_override)
        if path is None:
            raise HTTPException(
                status_code=400,
                detail="no script_path configured; pass {'script_path': ...} in the body or start the server with --script-path",
            )

        repo = _resolve_repo(request)
        if repo is None:
            raise HTTPException(
                status_code=400,
                detail="no repo_path configured on the server",
            )

        try:
            outcome = apply_decision(decision, script_path=path, repo=repo, run_id=run_id)
        except GitError as e:
            raise HTTPException(status_code=500, detail=f"git error: {e}") from e

        if not outcome.applied:
            raise HTTPException(
                status_code=422,
                detail=f"fix did not apply: {outcome.reason}",
            )

        store.resolve_decision(decision_id, approved=True)
        event = {
            "kind": "decision_applied",
            "run_id": run_id,
            "decision_id": decision_id,
            "template": decision.get("fix_template"),
            "commit": outcome.commit,
            "ts": __import__("time").time(),
        }
        store.append_event(run_id, "decision_applied", event)
        broker.publish(event)

        return {
            "status": "applied",
            "decision_id": decision_id,
            "commit": outcome.commit,
            "diff": outcome.diff,
        }

    @app.post("/api/runs/{run_id}/decisions/{decision_id}/reject")
    async def reject_decision(
        run_id: str,
        decision_id: int,
        request: Request,
    ) -> dict[str, Any]:
        store: EventStore = request.app.state.store
        broker: EventBroker = request.app.state.broker
        decision = store.get_decision(decision_id)
        if decision is None or decision.get("run_id") != run_id:
            raise HTTPException(status_code=404, detail=f"decision {decision_id} not found")
        if decision.get("approved") is not None:
            raise HTTPException(
                status_code=409,
                detail=f"decision {decision_id} already resolved",
            )
        store.resolve_decision(decision_id, approved=False)
        event = {
            "kind": "decision_rejected",
            "run_id": run_id,
            "decision_id": decision_id,
            "ts": __import__("time").time(),
        }
        store.append_event(run_id, "decision_rejected", event)
        broker.publish(event)
        return {"status": "rejected", "decision_id": decision_id}

    @app.post("/api/runs/{run_id}/decisions/{decision_id}/undo")
    async def undo_decision(
        run_id: str,
        decision_id: int,
        request: Request,
    ) -> dict[str, Any]:
        store: EventStore = request.app.state.store
        broker: EventBroker = request.app.state.broker
        decision = store.get_decision(decision_id)
        if decision is None or decision.get("run_id") != run_id:
            raise HTTPException(status_code=404, detail=f"decision {decision_id} not found")
        if decision.get("approved") != 1:
            raise HTTPException(
                status_code=409,
                detail="can only undo a previously-approved decision",
            )

        repo = _resolve_repo(request)
        if repo is None:
            raise HTTPException(
                status_code=400,
                detail="no repo_path configured on the server",
            )

        try:
            repo.reset_hard("HEAD~1")
        except GitError as e:
            raise HTTPException(status_code=500, detail=f"git reset failed: {e}") from e

        # Mark as un-applied (back to pending so the agent could re-propose).
        store.conn.execute(
            "UPDATE decisions SET approved = NULL, applied_at = NULL WHERE id = ?",
            (decision_id,),
        )
        event = {
            "kind": "decision_undone",
            "run_id": run_id,
            "decision_id": decision_id,
            "ts": __import__("time").time(),
        }
        store.append_event(run_id, "decision_undone", event)
        broker.publish(event)
        return {"status": "undone", "decision_id": decision_id}

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
