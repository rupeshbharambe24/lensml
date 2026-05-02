"""Unit tests for lensml.integrations.pytorch (the `lensml.watch()` API)."""

from __future__ import annotations

import io
import json

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import lensml
from lensml.core.transport import JSONLPublisher
from lensml.integrations.pytorch import WatchHandle, watch

# ---- helpers ----


class _MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(10, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


def _make_publisher() -> tuple[JSONLPublisher, io.StringIO]:
    """Return a JSONL publisher writing to an in-memory buffer."""
    buf = io.StringIO()
    return JSONLPublisher(stream=buf), buf


def _read_events(buf: io.StringIO) -> list[dict]:
    """Parse all JSONL lines emitted to the buffer."""
    return [json.loads(line) for line in buf.getvalue().splitlines() if line.strip()]


def _train_one_step(model: nn.Module, optimizer: torch.optim.Optimizer) -> torch.Tensor:
    optimizer.zero_grad(set_to_none=True)
    x = torch.randn(4, 10)
    y = torch.randint(0, 4, (4,))
    loss = nn.CrossEntropyLoss()(model(x), y)
    loss.backward()
    optimizer.step()
    return loss


# ---- public re-exports ----


def test_top_level_watch_re_export() -> None:
    """`lensml.watch` and `lensml.unwatch` are exposed at package root."""
    assert callable(lensml.watch)
    assert callable(lensml.unwatch)


# ---- WatchHandle / watch() basics ----


def test_watch_returns_handle_and_assigns_run_id() -> None:
    pub, _ = _make_publisher()
    model = _MLP()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    h = watch(model, opt, publisher=pub)
    assert isinstance(h, WatchHandle)
    assert h.run_id.startswith("run-")
    assert h.step == 0
    h.unwatch()


def test_watch_uses_explicit_run_id() -> None:
    pub, _ = _make_publisher()
    model = _MLP()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    h = watch(model, opt, run_id="my-run", publisher=pub)
    assert h.run_id == "my-run"
    h.unwatch()


def test_watch_emits_run_started_event() -> None:
    pub, buf = _make_publisher()
    model = _MLP()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    h = watch(model, opt, run_id="r", publisher=pub)
    events = _read_events(buf)
    kinds = [e["kind"] for e in events]
    assert "run_started" in kinds
    h.unwatch()


# ---- optimizer.step wrapping ----


def test_step_counter_advances_per_optimizer_step() -> None:
    pub, _ = _make_publisher()
    model = _MLP()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    h = watch(model, opt, publisher=pub, publish_every=1)

    for _ in range(3):
        _train_one_step(model, opt)

    assert h.step == 3
    h.unwatch()


def test_publish_every_throttles_emissions() -> None:
    """With publish_every=10 and 5 steps, no metric events should fire
    at boundaries other than step 0."""
    pub, buf = _make_publisher()
    model = _MLP()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    h = watch(model, opt, publisher=pub, publish_every=10)

    for _ in range(5):
        _train_one_step(model, opt)

    metric_events = [e for e in _read_events(buf) if e["kind"] == "metric"]
    # Step 0 fired at the first optimizer.step (since 0 % 10 == 0). No others.
    assert len(metric_events) == 1
    assert metric_events[0]["step"] == 0
    h.unwatch()


def test_published_metric_event_shape() -> None:
    pub, buf = _make_publisher()
    model = _MLP()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    h = watch(model, opt, run_id="r", publisher=pub, publish_every=1)

    _train_one_step(model, opt)

    metric_events = [e for e in _read_events(buf) if e["kind"] == "metric"]
    assert len(metric_events) >= 1
    e = metric_events[0]
    assert e["run_id"] == "r"
    assert e["step"] == 0
    assert "ts" in e
    assert "grad_l2" in e
    assert "lr" in e
    assert e["lr"] == pytest.approx(0.01)
    assert "activations" in e
    # MLP has one ReLU hooked
    assert "relu" in e["activations"]
    h.unwatch()


def test_handle_log_emits_user_event() -> None:
    pub, buf = _make_publisher()
    model = _MLP()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    h = watch(model, opt, run_id="r", publisher=pub)

    h.log(loss=0.5, accuracy=0.7)

    events = [e for e in _read_events(buf) if e["kind"] == "metric"]
    matching = [e for e in events if e.get("loss") == 0.5]
    assert len(matching) == 1
    assert matching[0]["accuracy"] == 0.7
    assert matching[0]["run_id"] == "r"
    assert matching[0]["step"] == 0
    h.unwatch()


# ---- teardown ----


def test_unwatch_restores_optimizer_step() -> None:
    pub, _ = _make_publisher()
    model = _MLP()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    original_class_step = type(opt).step
    h = watch(model, opt, publisher=pub)
    assert opt.step is not original_class_step  # wrapped (instance attr)
    h.unwatch()
    # After unwatch, no instance attribute — class method resolves.
    assert "step" not in opt.__dict__


def test_unwatch_removes_hooks() -> None:
    pub, _ = _make_publisher()
    model = _MLP()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    h = watch(model, opt, publisher=pub)

    # Forward populates the buffer
    _ = model(torch.randn(2, 10))
    assert len(h._buffer.activations) > 0

    h.unwatch()

    # After unwatch buffer is cleared and hooks no longer fire
    h._buffer.clear()
    _ = model(torch.randn(2, 10))
    assert h._buffer.activations == {}


def test_unwatch_idempotent() -> None:
    pub, _ = _make_publisher()
    model = _MLP()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    h = watch(model, opt, publisher=pub)
    h.unwatch()
    h.unwatch()  # second call must not raise


def test_context_manager_tears_down() -> None:
    pub, _ = _make_publisher()
    model = _MLP()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    with watch(model, opt, publisher=pub) as h:
        _train_one_step(model, opt)
        assert h.step == 1
    assert "step" not in opt.__dict__


def test_unwatch_emits_run_ended_event() -> None:
    pub, buf = _make_publisher()
    model = _MLP()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    h = watch(model, opt, run_id="r", publisher=pub)
    _train_one_step(model, opt)
    h.unwatch()
    events = _read_events(buf)
    ended = [e for e in events if e["kind"] == "run_ended"]
    assert len(ended) == 1
    assert ended[0]["step"] == 1


# ---- robustness ----


def test_telemetry_failure_does_not_crash_training() -> None:
    """A broken publisher must not propagate errors to the user's loop."""

    class _BrokenPublisher:
        def publish(self, event: dict) -> None:
            raise RuntimeError("network down")

        def close(self) -> None:
            pass

    model = _MLP()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    h = watch(model, opt, publisher=_BrokenPublisher(), publish_every=1)

    # Should not raise even though publisher always fails
    for _ in range(3):
        _train_one_step(model, opt)
    h.unwatch()


def test_works_with_adam_optimizer() -> None:
    """Wrapping must work for optimizers other than SGD."""
    pub, _ = _make_publisher()
    model = _MLP()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    h = watch(model, opt, publisher=pub, publish_every=1)
    for _ in range(2):
        _train_one_step(model, opt)
    assert h.step == 2
    h.unwatch()


def test_dataloader_argument_accepted() -> None:
    """The dataloader arg is accepted for API compatibility (currently unused)."""
    pub, _ = _make_publisher()
    model = _MLP()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    ds = TensorDataset(torch.randn(8, 10), torch.zeros(8, dtype=torch.long))
    dl = DataLoader(ds, batch_size=4)
    h = watch(model, opt, dl, publisher=pub)
    assert h.run_id.startswith("run-")
    h.unwatch()


def test_top_level_unwatch_function() -> None:
    """`lensml.unwatch(handle)` is equivalent to `handle.unwatch()`."""
    pub, _ = _make_publisher()
    model = _MLP()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    h = watch(model, opt, publisher=pub)
    lensml.unwatch(h)
    assert "step" not in opt.__dict__
