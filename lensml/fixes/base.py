"""FixTemplate ABC — base class for all libcst transformers.

Each concrete fix template:
    1. Accepts structured params (e.g., {layer_name: "fc1", new_lr: 0.001}).
    2. Locates the target node(s) via libcst.matchers.
    3. Returns a unified diff against the original source.
    4. Has a unit test on a known input (plan §8 Day 11).

Per plan §5.2, only the agent's `propose_fix` tool calls into this layer,
and only via the server (not the agent process directly).
"""

from __future__ import annotations

# Implemented Day 11.
