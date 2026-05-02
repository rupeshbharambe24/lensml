"""LowerLearningRate — fix for LR-too-high oscillating loss (plan §4 row 2).

Multiplies any ``lr=L`` keyword argument (in optimizer construction or any
other call) by a configurable factor (default 0.3 per plan §4).

Behavior on numeric literals: the value is replaced with the computed
literal (e.g. ``lr=0.01`` → ``lr=0.003``). On non-literal expressions
(``lr=base_lr`` or ``lr=cfg["lr"]``), wraps in a ``BinaryOperation``
producing ``lr=base_lr * 0.3``.
"""

from __future__ import annotations

from typing import ClassVar

import libcst as cst
import libcst.matchers as m

from lensml.fixes.base import FixResult, FixTemplate


class _LRScaler(cst.CSTTransformer):
    def __init__(self, factor: float) -> None:
        super().__init__()
        self.factor = factor
        self.applied = False

    def leave_Arg(self, original_node: cst.Arg, updated_node: cst.Arg) -> cst.Arg:
        kw = updated_node.keyword
        if not isinstance(kw, cst.Name) or kw.value != "lr":
            return updated_node

        new_value = self._scale(updated_node.value)
        if new_value is None:
            return updated_node
        self.applied = True
        return updated_node.with_changes(value=new_value)

    def _scale(self, expr: cst.BaseExpression) -> cst.BaseExpression | None:
        # Replace numeric literal in place.
        if m.matches(expr, m.Float() | m.Integer()):
            try:
                old_val = float(expr.value)  # type: ignore[attr-defined]
            except (TypeError, ValueError):
                return None
            new_val = old_val * self.factor
            return cst.Float(value=repr(new_val))

        # Wrap expression in (expr * factor).
        return cst.BinaryOperation(
            left=expr,
            operator=cst.Multiply(
                whitespace_before=cst.SimpleWhitespace(" "),
                whitespace_after=cst.SimpleWhitespace(" "),
            ),
            right=cst.Float(value=repr(self.factor)),
        )


class LowerLearningRate(FixTemplate):
    name: ClassVar[str] = "LowerLearningRate"

    def __init__(self, factor: float = 0.3) -> None:
        if factor <= 0:
            raise ValueError(f"factor must be > 0, got {factor!r}")
        self.factor = factor

    def apply(self, source: str) -> FixResult:
        try:
            tree = cst.parse_module(source)
        except Exception as e:
            return FixResult(
                applied=False,
                template=self.name,
                old_source=source,
                reason=f"parse error: {type(e).__name__}: {e}",
            )
        scaler = _LRScaler(self.factor)
        new_tree = tree.visit(scaler)
        if not scaler.applied:
            return FixResult(
                applied=False,
                template=self.name,
                old_source=source,
                reason="no `lr=` keyword argument found",
            )
        return FixResult(
            applied=True,
            template=self.name,
            old_source=source,
            new_source=new_tree.code,
        )
