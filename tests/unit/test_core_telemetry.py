"""Unit tests for lensml.core.telemetry."""

from __future__ import annotations

import torch
import torch.nn as nn

from lensml.core.hooks import TelemetryBuffer, install_capture_hooks, remove_hooks
from lensml.core.telemetry import (
    SAMPLE_SIZE,
    _strided_sample,
    activation_stats,
    global_grad_l2,
    per_layer_grad_l2,
)

# ---- _strided_sample ----


def test_strided_sample_short_tensor_returns_full() -> None:
    t = torch.arange(100.0)
    s = _strided_sample(t, max_elements=200)
    assert torch.equal(s, t)


def test_strided_sample_exact_size() -> None:
    t = torch.arange(100.0)
    s = _strided_sample(t, max_elements=100)
    assert torch.equal(s, t)


def test_strided_sample_long_tensor_capped() -> None:
    t = torch.arange(10_000.0)
    s = _strided_sample(t, max_elements=1000)
    assert s.numel() <= 1000


def test_strided_sample_default_size() -> None:
    t = torch.randn(100_000)
    s = _strided_sample(t)
    assert s.numel() <= SAMPLE_SIZE


def test_strided_sample_preserves_distribution() -> None:
    """On a random tensor, sample mean should be close to full mean."""
    torch.manual_seed(0)
    t = torch.randn(100_000)
    full_mean = float(t.mean())
    sample_mean = float(_strided_sample(t, max_elements=1024).mean())
    assert abs(full_mean - sample_mean) < 0.2


def test_strided_sample_multidim_flattened() -> None:
    t = torch.randn(8, 16, 32)
    s = _strided_sample(t, max_elements=64)
    assert s.dim() == 1
    assert s.numel() <= 64


# ---- activation_stats ----


def test_activation_stats_returns_python_floats() -> None:
    buf = TelemetryBuffer()
    buf.activations["x"] = torch.tensor([1.0, 2.0, 0.0, 3.0])
    stats = activation_stats(buf)
    assert "x" in stats
    assert isinstance(stats["x"]["mean"], float)
    assert isinstance(stats["x"]["dead_fraction"], float)


def test_activation_stats_dead_fraction_exact() -> None:
    """4 zeros / 8 = 0.5 dead fraction."""
    buf = TelemetryBuffer()
    buf.activations["x"] = torch.tensor([0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0])
    stats = activation_stats(buf)
    assert stats["x"]["dead_fraction"] == 0.5


def test_activation_stats_empty_buffer() -> None:
    buf = TelemetryBuffer()
    stats = activation_stats(buf)
    assert stats == {}


# ---- global_grad_l2 ----


def test_global_grad_l2_zero_when_no_grad() -> None:
    model = nn.Linear(3, 2)
    assert global_grad_l2(model) == 0.0


def test_global_grad_l2_matches_manual() -> None:
    """Manual sqrt-of-sum-of-squared-norms should match global_grad_l2."""
    torch.manual_seed(0)
    model = nn.Linear(3, 2)
    x = torch.randn(4, 3)
    y = torch.randint(0, 2, (4,))
    loss = nn.CrossEntropyLoss()(model(x), y)
    loss.backward()

    manual = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
    assert abs(global_grad_l2(model) - manual) < 1e-5


def test_global_grad_l2_empty_model() -> None:
    """A module with no parameters returns 0.0."""

    class _Empty(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    assert global_grad_l2(_Empty()) == 0.0


# ---- per_layer_grad_l2 ----


def test_per_layer_grad_l2() -> None:
    torch.manual_seed(0)
    model = nn.Linear(3, 2)
    x = torch.randn(4, 3)
    y = torch.randint(0, 2, (4,))
    loss = nn.CrossEntropyLoss()(model(x), y)
    loss.backward()

    grads = per_layer_grad_l2(model)
    assert "weight" in grads
    assert "bias" in grads
    assert grads["weight"] > 0
    assert grads["bias"] > 0


def test_per_layer_grad_l2_skips_no_grad() -> None:
    model = nn.Linear(3, 2)
    grads = per_layer_grad_l2(model)
    assert grads == {}


# ---- end-to-end integration ----


def test_integration_hooks_then_stats() -> None:
    """End-to-end: install hooks, forward, compute stats."""
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
    buf = TelemetryBuffer()
    handles = install_capture_hooks(model, buf)

    _ = model(torch.randn(8, 10))

    stats = activation_stats(buf)
    assert len(stats) == 1
    relu_stats = next(iter(stats.values()))
    # ReLU output is non-negative
    assert relu_stats["mean"] >= 0.0
    assert 0.0 <= relu_stats["dead_fraction"] <= 1.0

    remove_hooks(handles)


def test_integration_with_backward() -> None:
    """Forward + backward + activation stats + global grad L2."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
    buf = TelemetryBuffer()
    handles = install_capture_hooks(model, buf)

    x = torch.randn(8, 10)
    y = torch.randint(0, 5, (8,))
    loss = nn.CrossEntropyLoss()(model(x), y)
    loss.backward()

    stats = activation_stats(buf)
    grad_l2 = global_grad_l2(model)

    assert len(stats) == 1
    assert grad_l2 > 0.0

    remove_hooks(handles)
