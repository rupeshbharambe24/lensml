"""REST API endpoints — historical placeholder.

Plan §3.2's directory tree listed this file as the home for REST
endpoints, but during the build the FastAPI app factory in
``lensml.server.app`` grew to host them inline (the closure captures
``app.state.store`` / ``broker`` cleanly without an extra module).

Where the endpoints actually live now:

    GET  /api/runs                                      → app.list_runs
    GET  /api/runs/{id}                                  → app.get_run
    GET  /api/runs/{id}/events                           → app.get_events
    GET  /api/runs/{id}/decisions                        → app.list_decisions
    GET  /api/runs/{id}/decisions/{decision_id}          → app.get_decision
    POST /api/runs/{id}/decisions/{decision_id}/approve  → app.approve_decision
    POST /api/runs/{id}/decisions/{decision_id}/reject   → app.reject_decision
    POST /api/runs/{id}/decisions/{decision_id}/undo     → app.undo_decision
    POST /api/runs/{id}/_test_publish                    → app.test_publish
    GET  /api/runs/{id}/metrics                          → app.metrics_stream (SSE)
    GET  /api/privacy        POST /api/privacy           → app.{get,set}_privacy
    POST /api/privacy/canned                             → app.privacy_canned

This module is kept (rather than deleted) so anyone navigating from
plan §3.2's directory tree lands here and gets pointed at the right
file.
"""

from __future__ import annotations
