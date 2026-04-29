"""Forward-hook installation for activation telemetry capture.

Plan refs:
    §2  — `register_forward_hook` is stable in PyTorch 2.x. The deprecated
          `register_backward_hook` must be avoided. Reading `param.grad`
          after `loss.backward()` is the canonical low-overhead path for
          per-parameter gradient stats (see lensml.core.telemetry).
    §3.1 — telemetry ticks at 10-50 Hz; hooks must stay in the hot path's
          microseconds budget.
    §4   — hook stats feed `dead_relu` (row 3), `inspect_activations`
          tool (§5.2), and the activation-mean signal for several
          detectors.

Spike 0.4 validated this architecture (notes/hooks-spike-2026-04-29.md):
the hook does only `out.detach()` (tensor view, no copy) and stores a
reference; reductions happen in the cold path (telemetry.publish_*).
Measured: 0.18 ms/step absolute on both CPU and GPU.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch.utils.hooks import RemovableHandle


# Activation modules whose outputs feed v1 detectors. Other module types
# (Conv2d, Linear, BatchNorm, etc.) are inspected on-demand by agent tools
# per plan §5.2 — not captured per-step.
ACTIVATION_TYPES: tuple[type, ...] = (
    nn.ReLU,
    nn.GELU,
    nn.Sigmoid,
    nn.Tanh,
    nn.LeakyReLU,
    nn.ELU,
    nn.SiLU,
)


class TelemetryBuffer:
    """Hot-path storage for the latest forward output per hooked module.

    The hook stored on each module is intentionally minimal: a single
    `isinstance` check + `out.detach()` (which is a view, no data copy)
    + a dict assignment. Reductions and Python conversions are deferred
    to `lensml.core.telemetry.activation_stats`, which runs at the
    publish rate (every N steps), not every forward.

    Memory: at any moment, the buffer holds references to the latest
    forward outputs of all hooked modules. For a small CNN that is a
    few MB; for ResNet-18 ~30 MB. The references are overwritten on
    each forward pass, so memory is bounded to one forward's worth.
    """

    __slots__ = ("activations",)

    def __init__(self) -> None:
        self.activations: dict[str, torch.Tensor] = {}

    def make_hook(self, name: str):
        def hook(_module: nn.Module, _inp: object, out: object) -> None:
            if isinstance(out, torch.Tensor):
                self.activations[name] = out.detach()

        return hook

    def clear(self) -> None:
        self.activations.clear()


def install_capture_hooks(
    model: nn.Module,
    buf: TelemetryBuffer,
    activation_types: tuple[type, ...] = ACTIVATION_TYPES,
) -> list[RemovableHandle]:
    """Install forward hooks on every module that matches `activation_types`.

    Returns the list of `RemovableHandle` objects. Caller is responsible
    for invoking `.remove()` on each at teardown — `remove_hooks(handles)`
    is the convenience helper. The user-facing `lensml.watch()` API wraps
    this in a context manager.
    """
    handles: list[RemovableHandle] = []
    for name, module in model.named_modules():
        if isinstance(module, activation_types):
            handles.append(module.register_forward_hook(buf.make_hook(name)))
    return handles


def remove_hooks(handles: list[RemovableHandle]) -> None:
    """Remove all installed hooks. Safe to call multiple times."""
    for h in handles:
        h.remove()
    handles.clear()
