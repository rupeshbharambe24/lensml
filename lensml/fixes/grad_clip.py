"""AddGradientClipping — fix for exploding gradients (plan §4 row 1).

Inserts ``torch.nn.utils.clip_grad_norm_(<model>.parameters(), max_norm=<N>)``
immediately AFTER any ``<x>.backward()`` statement that is directly followed
by an ``<y>.step()`` statement in the same block.

Assumes ``torch`` is already imported (true for any PyTorch training script
worth instrumenting). For v1 we don't auto-add imports — that's a v0.2 polish
once we standardize on `libcst.codemod.visitors.AddImportsVisitor`.
"""

from __future__ import annotations

from typing import ClassVar

import libcst as cst
import libcst.matchers as m

from lensml.fixes.base import FixResult, FixTemplate


def _is_backward_stmt(stmt: cst.BaseStatement) -> bool:
    return m.matches(
        stmt,
        m.SimpleStatementLine(
            body=[m.Expr(value=m.Call(func=m.Attribute(attr=m.Name("backward"))))]
        ),
    )


def _is_step_stmt(stmt: cst.BaseStatement) -> bool:
    return m.matches(
        stmt,
        m.SimpleStatementLine(body=[m.Expr(value=m.Call(func=m.Attribute(attr=m.Name("step"))))]),
    )


def _is_clip_stmt(stmt: cst.BaseStatement) -> bool:
    """Detect an existing clip_grad_norm_ call so we don't double-apply."""
    return m.matches(
        stmt,
        m.SimpleStatementLine(
            body=[m.Expr(value=m.Call(func=m.Attribute(attr=m.Name("clip_grad_norm_"))))]
        ),
    )


class _Inserter(cst.CSTTransformer):
    def __init__(self, max_norm: float, model_name: str) -> None:
        super().__init__()
        self.max_norm = max_norm
        self.model_name = model_name
        self.applied = False

    def leave_IndentedBlock(
        self,
        original_node: cst.IndentedBlock,
        updated_node: cst.IndentedBlock,
    ) -> cst.IndentedBlock:
        body = list(updated_node.body)
        if len(body) < 2:
            return updated_node

        new_body: list[cst.BaseStatement] = []
        for i, stmt in enumerate(body):
            new_body.append(stmt)
            if not _is_backward_stmt(stmt):
                continue
            # Look ahead one step for *.step()
            if i + 1 >= len(body):
                continue
            next_stmt = body[i + 1]
            if not _is_step_stmt(next_stmt):
                continue
            # Skip if already clipped (clip already present between them)
            if i + 2 < len(body) and _is_clip_stmt(body[i + 1]):
                continue
            clip_code = (
                f"torch.nn.utils.clip_grad_norm_("
                f"{self.model_name}.parameters(), max_norm={self.max_norm})"
            )
            clip_stmt = cst.parse_statement(clip_code)
            new_body.append(clip_stmt)
            self.applied = True

        if not self.applied:
            return updated_node
        return updated_node.with_changes(body=tuple(new_body))


class AddGradientClipping(FixTemplate):
    name: ClassVar[str] = "AddGradientClipping"

    def __init__(self, max_norm: float = 1.0, model_name: str = "model") -> None:
        self.max_norm = max_norm
        self.model_name = model_name

    def apply(self, source: str) -> FixResult:
        try:
            tree = cst.parse_module(source)
        except Exception as e:  # libcst raises various subclasses
            return FixResult(
                applied=False,
                template=self.name,
                old_source=source,
                reason=f"parse error: {type(e).__name__}: {e}",
            )

        inserter = _Inserter(self.max_norm, self.model_name)
        new_tree = tree.visit(inserter)
        if not inserter.applied:
            return FixResult(
                applied=False,
                template=self.name,
                old_source=source,
                reason="no `.backward()` → `.step()` pattern found",
            )
        return FixResult(
            applied=True,
            template=self.name,
            old_source=source,
            new_source=new_tree.code,
        )
