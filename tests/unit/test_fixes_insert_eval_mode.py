"""Unit tests for InsertEvalMode (plan §4 row 6)."""

from __future__ import annotations

import libcst as cst

from lensml.fixes.insert_eval_mode import InsertEvalMode


def _round_trip(src: str) -> None:
    """Sanity: result re-parses cleanly."""
    cst.parse_module(src)


def test_inserts_eval_and_train_in_validate_function() -> None:
    src = """
def validate(model, loader):
    for batch in loader:
        out = model(batch)
"""
    out = InsertEvalMode().apply(src)
    assert out.applied is True
    assert "model.eval()" in out.new_source
    assert "model.train()" in out.new_source
    assert out.new_source.index("model.eval()") < out.new_source.index("for batch")
    assert out.new_source.index("for batch") < out.new_source.index("model.train()")
    _round_trip(out.new_source)


def test_recognizes_evaluate_function_name() -> None:
    src = """
def evaluate(net, loader):
    pass
"""
    out = InsertEvalMode(model_name="net").apply(src)
    assert out.applied is True
    assert "net.eval()" in out.new_source
    assert "net.train()" in out.new_source


def test_recognizes_validation_step() -> None:
    src = """
def validation_step(self, batch, batch_idx):
    out = self.model(batch)
"""
    # The function name validation_step matches; transformer prepends
    # `<model_name>.eval()` regardless of whether that name is referenced
    # inside the body — the agent picks the right model name from context.
    out = InsertEvalMode(model_name="self.model").apply(src)
    assert out.applied is True
    assert "self.model.eval()" in out.new_source
    assert "self.model.train()" in out.new_source


def test_skips_unrelated_function() -> None:
    src = """
def train_one_epoch(model, loader):
    pass
"""
    out = InsertEvalMode().apply(src)
    assert out.applied is False
    assert "no validation" in (out.reason or "")


def test_idempotent_when_eval_already_first() -> None:
    src = """
def validate(model, loader):
    model.eval()
    for batch in loader:
        out = model(batch)
    model.train()
"""
    out = InsertEvalMode().apply(src)
    # Already wrapped — nothing to add.
    assert out.applied is False


def test_appends_train_when_only_eval_present() -> None:
    src = """
def validate(model, loader):
    model.eval()
    for batch in loader:
        out = model(batch)
"""
    out = InsertEvalMode().apply(src)
    assert out.applied is True
    assert out.new_source.count("model.eval()") == 1
    assert "model.train()" in out.new_source


def test_parse_error_returns_unapplied() -> None:
    src = "def broken("
    out = InsertEvalMode().apply(src)
    assert out.applied is False
    assert "parse error" in (out.reason or "")


def test_diff_property_when_applied() -> None:
    src = """
def validate(model, loader):
    pass
"""
    out = InsertEvalMode().apply(src)
    assert out.applied is True
    assert "+    model.eval()" in out.diff
    assert "+    model.train()" in out.diff
