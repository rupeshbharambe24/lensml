"""Server-Sent Events endpoints.

Plan §3.1: SSE via sse-starlette is the right transport for one-way
server→browser streams of training metrics. Native EventSource auto-reconnects.
WebSocket would be overkill.

Endpoints (plan §3.2):
    GET /api/runs/{id}/metrics   live metric ticks (10–50/sec)
    GET /api/runs/{id}/agent     agent reasoning trace (token-by-token)
"""

from __future__ import annotations

# Implemented Day 3.
