"""lensml.server — local FastAPI server.

Per plan §3.1: server holds long-lived state (event log, agent context,
git refs). Default bind: 127.0.0.1 (do NOT expose externally; see SECURITY.md).

Module map (plan §3.2):
    app.py        FastAPI app
    sse.py        /api/runs/{id}/metrics, /api/runs/{id}/agent
    api.py        REST endpoints (approve, reject, undo)
    git.py        branch / commit / reset
    checkpoint.py save/load model + opt + scheduler + RNG + sampler
    store.py      SQLite event log
"""
