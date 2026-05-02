"""Unit tests for AddLRScheduler (plan §3.2 templates)."""

from __future__ import annotations

import libcst as cst

from lensml.fixes.add_lr_scheduler import AddLRScheduler


def _round_trip(src: str) -> None:
    cst.parse_module(src)


def test_inserts_scheduler_assignment_after_optimizer() -> None:
    src = """
import torch
model = MyNet()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
"""
    out = AddLRScheduler(t_max=100).apply(src)
    assert out.applied is True
    assert "scheduler = torch.optim.lr_scheduler.CosineAnnealingLR" in out.new_source
    assert "T_max=100" in out.new_source
    _round_trip(out.new_source)


def test_inserts_scheduler_step_after_optimizer_step() -> None:
    src = """
optimizer = torch.optim.SGD([], lr=0.01)
for batch in loader:
    optimizer.step()
"""
    out = AddLRScheduler().apply(src)
    assert out.applied is True
    assert "scheduler.step()" in out.new_source
    # Order: optimizer.step() then scheduler.step()
    opt_idx = out.new_source.index("optimizer.step()")
    sched_step_idx = out.new_source.index("scheduler.step()")
    assert opt_idx < sched_step_idx


def test_skips_when_scheduler_already_present() -> None:
    src = """
optimizer = torch.optim.SGD([], lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
"""
    out = AddLRScheduler().apply(src)
    # Assignment is already there, but optimizer.step() pattern isn't here
    # either, so nothing applies.
    assert out.applied is False


def test_skips_step_when_scheduler_step_already_chained() -> None:
    src = """
optimizer = torch.optim.SGD([], lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
for batch in loader:
    optimizer.step()
    scheduler.step()
"""
    out = AddLRScheduler().apply(src)
    assert out.applied is False


def test_custom_optimizer_and_scheduler_names() -> None:
    src = """
opt = torch.optim.SGD([], lr=0.01)
opt.step()
"""
    out = AddLRScheduler(optimizer_name="opt", scheduler_name="sched").apply(src)
    assert out.applied is True
    assert "sched = torch.optim.lr_scheduler.CosineAnnealingLR" in out.new_source
    assert "sched.step()" in out.new_source


def test_no_optimizer_no_change() -> None:
    src = "print('hello')\n"
    out = AddLRScheduler().apply(src)
    assert out.applied is False
    assert "no `optimizer = ...`" in (out.reason or "")


def test_parse_error_returns_unapplied() -> None:
    out = AddLRScheduler().apply("def broken(")
    assert out.applied is False


def test_diff_when_applied() -> None:
    src = "optimizer = torch.optim.SGD([], lr=0.01)\n"
    out = AddLRScheduler().apply(src)
    assert out.applied is True
    assert "+scheduler" in out.diff


def test_t_max_parameter_renders() -> None:
    src = "optimizer = torch.optim.Adam([], lr=0.001)\n"
    out = AddLRScheduler(t_max=42).apply(src)
    assert out.applied is True
    assert "T_max=42" in out.new_source
