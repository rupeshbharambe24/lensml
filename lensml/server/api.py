"""REST API endpoints (separate concern from SSE streams).

Plan §3.1: approval clicks go via REST POST.

Endpoints:
    POST /api/runs/{id}/decisions/{decision_id}/approve
    POST /api/runs/{id}/decisions/{decision_id}/reject
    POST /api/runs/{id}/decisions/{decision_id}/undo
    GET  /api/runs/{id}/decisions
    GET  /api/runs/{id}/status
"""

from __future__ import annotations

# Implemented Day 12.
