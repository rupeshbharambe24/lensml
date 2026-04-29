"""Forward and gradient hook installers.

Plan refs:
    §2 — `register_forward_hook` and `register_full_backward_hook` are stable
         in PyTorch 2.x. The deprecated `register_backward_hook` must be
         avoided. Reading `param.grad` after `loss.backward()` is the
         canonical, low-overhead way to get per-parameter gradient stats.
    §8 Day 1–2 — telemetry layer build.
"""

from __future__ import annotations

# Implemented Day 1–2.
