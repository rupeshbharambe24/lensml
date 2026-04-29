"""Cold-path metric extraction from captured activations and gradient state.

The publish layer of plan §3.1's two-tier telemetry design. Runs every N
training steps (the user-facing API tunes N to match plan §3.1's 10-50 Hz
tick rate). Produces structured Python dicts that the transport layer
ships to the FastAPI server.

Validated by spike 0.4 (notes/hooks-spike-2026-04-29.md):
    - Strided sub-sampling (1024 elements) bounds reduction work regardless
      of tensor size. On reasonably-distributed activations: mean error
      ~5%, dead_fraction error ~0.05% — adequate for trend-based detectors
      (plan §4 thresholds use rolling means, not exact per-step values).
    - Single global grad-L2 is what `exploding_grad` (plan §4 row 1) needs.
      Per-parameter L2 (for `vanishing_grad`, plan §4 row 8) is on-demand
      only — not in the per-step path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from lensml.core.hooks import TelemetryBuffer


SAMPLE_SIZE: int = 1024


def _strided_sample(t: torch.Tensor, max_elements: int = SAMPLE_SIZE) -> torch.Tensor:
    """Strided 1-D sample of `t`, capped at `max_elements`.

    For tensors smaller than `max_elements`, returns the full flattened tensor.
    For larger tensors, returns every k-th element where k = numel / max_elements.
    """
    flat = t.reshape(-1)
    n = flat.numel()
    if n <= max_elements:
        return flat
    # Strided slice may yield up to `max_elements + (n % stride != 0)` samples
    # due to integer-division rounding; the trailing slice tightens the cap.
    return flat[:: n // max_elements][:max_elements]


def activation_stats(
    buf: TelemetryBuffer,
    sample_size: int = SAMPLE_SIZE,
) -> dict[str, dict[str, float]]:
    """Compute mean and dead_fraction per captured activation.

    Returns a nested dict: ``{module_name: {"mean": float, "dead_fraction": float}}``.

    The result is plain Python floats — `.item()` conversion is performed
    here (in the cold path), not inside the hook (the hot path).
    """
    out: dict[str, dict[str, float]] = {}
    for name, t in buf.activations.items():
        sample = _strided_sample(t, sample_size)
        mean_val = float(sample.mean())
        dead_val = float(sample.eq(0).count_nonzero().float() / sample.numel())
        out[name] = {"mean": mean_val, "dead_fraction": dead_val}
    return out


def global_grad_l2(model: nn.Module) -> float:
    """Single global L2 of all parameters' gradients (plan §4 row 1 signal).

    Computed as ``sqrt(sum(p.grad.norm()**2 for p in model.parameters()))``.
    Reads `param.grad` directly post-`loss.backward()` — no backward hooks.

    Returns 0.0 if no parameter has a gradient yet.
    """
    params = list(model.parameters())
    if not params:
        return 0.0
    device = params[0].device
    total_sq = torch.zeros((), device=device)
    for p in params:
        if p.grad is not None:
            total_sq = total_sq + p.grad.norm() ** 2
    return float(total_sq.sqrt())


def per_layer_grad_l2(model: nn.Module) -> dict[str, float]:
    """Per-parameter gradient L2 norms.

    Used by the `inspect_gradient_flow` agent tool (plan §5.2) and the
    `vanishing_grad` detector (plan §4 row 8). On-demand only — not
    called every step.
    """
    out: dict[str, float] = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            out[name] = float(p.grad.norm())
    return out
