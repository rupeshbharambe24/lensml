"""Unit tests for KaimingInit (plan §4 row 7)."""

from __future__ import annotations

import libcst as cst

from lensml.fixes.kaiming_init import KaimingInit


def _round_trip(src: str) -> None:
    cst.parse_module(src)


def test_inserts_apply_call_after_top_level_assignment() -> None:
    src = """
import torch
model = MyNet()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
"""
    out = KaimingInit().apply(src)
    assert out.applied is True
    assert "model.apply(" in out.new_source
    assert "kaiming_normal_" in out.new_source
    assert out.new_source.index("model = MyNet()") < out.new_source.index("model.apply(")
    assert out.new_source.index("model.apply(") < out.new_source.index("optimizer =")
    _round_trip(out.new_source)


def test_inserts_apply_call_inside_function_body() -> None:
    src = """
def build():
    model = MyNet()
    return model
"""
    out = KaimingInit().apply(src)
    assert out.applied is True
    assert "model.apply(" in out.new_source


def test_skips_when_apply_already_present() -> None:
    src = """
model = MyNet()
model.apply(my_init_fn)
"""
    out = KaimingInit().apply(src)
    assert out.applied is False


def test_no_assignment_no_change() -> None:
    src = """
import torch
print('hello')
"""
    out = KaimingInit().apply(src)
    assert out.applied is False
    assert "no `model" in (out.reason or "")


def test_custom_model_name() -> None:
    src = """
net = MyNet()
"""
    out = KaimingInit(model_name="net").apply(src)
    assert out.applied is True
    assert "net.apply(" in out.new_source


def test_only_first_assignment_gets_apply() -> None:
    """If the user reassigns `model` later, only the first construction site
    is wrapped — the later assignment is presumed intentional."""
    src = """
model = MyNet()
model = wrap(model)
"""
    out = KaimingInit().apply(src)
    assert out.applied is True
    assert out.new_source.count("model.apply(") == 1


def test_parse_error_returns_unapplied() -> None:
    out = KaimingInit().apply("def broken(")
    assert out.applied is False


def test_diff_when_applied() -> None:
    src = "model = MyNet()\n"
    out = KaimingInit().apply(src)
    assert out.applied is True
    assert "+model.apply(" in out.diff
