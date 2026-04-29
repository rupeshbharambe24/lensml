"""Unit tests for lensml.core.hooks."""

from __future__ import annotations

import torch
import torch.nn as nn

from lensml.core.hooks import (
    ACTIVATION_TYPES,
    TelemetryBuffer,
    install_capture_hooks,
    remove_hooks,
)


class _MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


def test_telemetry_buffer_starts_empty() -> None:
    buf = TelemetryBuffer()
    assert buf.activations == {}


def test_buffer_clear() -> None:
    buf = TelemetryBuffer()
    buf.activations["x"] = torch.zeros(4)
    buf.clear()
    assert buf.activations == {}


def test_capture_hooks_only_on_activations() -> None:
    """Hooks fire only on modules in ACTIVATION_TYPES, not on Linear etc."""
    model = _MLP()
    buf = TelemetryBuffer()
    handles = install_capture_hooks(model, buf)

    _ = model(torch.randn(4, 10))

    assert set(buf.activations.keys()) == {"relu"}
    assert buf.activations["relu"].shape == (4, 20)

    remove_hooks(handles)


def test_remove_hooks_stops_capture() -> None:
    model = _MLP()
    buf = TelemetryBuffer()
    handles = install_capture_hooks(model, buf)

    _ = model(torch.randn(4, 10))
    assert "relu" in buf.activations

    buf.clear()
    remove_hooks(handles)

    _ = model(torch.randn(4, 10))
    assert buf.activations == {}


def test_remove_hooks_idempotent() -> None:
    model = _MLP()
    buf = TelemetryBuffer()
    handles = install_capture_hooks(model, buf)
    remove_hooks(handles)
    remove_hooks(handles)


def test_multiple_activation_types() -> None:
    """Should hook ReLU, GELU, Sigmoid, etc. — anything in ACTIVATION_TYPES."""
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 8),
        nn.GELU(),
        nn.Linear(8, 8),
        nn.Sigmoid(),
    )
    buf = TelemetryBuffer()
    handles = install_capture_hooks(model, buf)

    _ = model(torch.randn(2, 4))

    assert len(buf.activations) == 3

    remove_hooks(handles)


def test_custom_activation_types() -> None:
    """Override which module types get hooked."""
    model = _MLP()
    buf = TelemetryBuffer()
    handles = install_capture_hooks(model, buf, activation_types=(nn.Linear,))

    _ = model(torch.randn(4, 10))

    assert set(buf.activations.keys()) == {"fc1", "fc2"}

    remove_hooks(handles)


def test_capture_overwrites_per_forward() -> None:
    """Each forward replaces the activations rather than accumulating."""
    model = _MLP()
    buf = TelemetryBuffer()
    handles = install_capture_hooks(model, buf)

    _ = model(torch.randn(4, 10))
    first = buf.activations["relu"]

    _ = model(torch.randn(4, 10))
    second = buf.activations["relu"]

    assert not torch.equal(first, second)

    remove_hooks(handles)


def test_activation_types_export() -> None:
    """ACTIVATION_TYPES should include the standard set."""
    assert nn.ReLU in ACTIVATION_TYPES
    assert nn.GELU in ACTIVATION_TYPES
    assert nn.Sigmoid in ACTIVATION_TYPES
    assert nn.Tanh in ACTIVATION_TYPES
    assert nn.LeakyReLU in ACTIVATION_TYPES
