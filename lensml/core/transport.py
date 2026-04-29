"""ZMQ PUB/SUB transport between trainer subprocess and FastAPI server.

Plan refs:
    §3.1 — pyzmq PUB/SUB is ~5 lines, survives either side restarting,
           microsecond latency. Socket: `ipc://lensml-{pid}.sock`.
    §3.1 — Inproc fallback for testing: stdin/stdout JSONL when ZMQ
           is unavailable.
"""

from __future__ import annotations

# Implemented Day 2.
