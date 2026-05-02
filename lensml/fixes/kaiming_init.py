"""KaimingInit — fix for bad initialization (plan §4 row 7).

Inserts a ``<model>.apply(...)`` call immediately after the first
assignment that constructs the user's model. The applied function is
inlined as a lambda (no top-level helper insertion needed) and uses
``torch.nn.init.kaiming_normal_`` for ``Linear`` and ``Conv2d`` layers.

Pattern matched (the assignment that triggers insertion):

    <model_name> = SomeClass(...)

After:

    <model_name> = SomeClass(...)
    <model_name>.apply(<init_lambda>)

Idempotent: skips if a ``<model_name>.apply(...)`` call already
follows the assignment.
"""

from __future__ import annotations

from typing import ClassVar

import libcst as cst
import libcst.matchers as m

from lensml.fixes.base import FixResult, FixTemplate

_INIT_LAMBDA = (
    "lambda _m: torch.nn.init.kaiming_normal_(_m.weight) "
    "if isinstance(_m, (torch.nn.Linear, torch.nn.Conv2d)) else None"
)


def _is_model_assignment(stmt: cst.BaseStatement, model_name: str) -> bool:
    """True if `stmt` is `<model_name> = <Call>(...)`."""
    return m.matches(
        stmt,
        m.SimpleStatementLine(
            body=[
                m.Assign(
                    targets=[m.AssignTarget(target=m.Name(model_name))],
                    value=m.Call(),
                )
            ]
        ),
    )


def _is_apply_call(stmt: cst.BaseStatement, model_name: str) -> bool:
    """True if `stmt` is `<model_name>.apply(...)`."""
    return m.matches(
        stmt,
        m.SimpleStatementLine(
            body=[
                m.Expr(
                    value=m.Call(func=m.Attribute(value=m.Name(model_name), attr=m.Name("apply")))
                )
            ]
        ),
    )


class _Inserter(cst.CSTTransformer):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.model_name = model_name
        self.applied = False

    def leave_IndentedBlock(
        self,
        original_node: cst.IndentedBlock,
        updated_node: cst.IndentedBlock,
    ) -> cst.IndentedBlock:
        return self._maybe_insert_after_assignment(updated_node)

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        # Top-level statements live directly on Module, not in an
        # IndentedBlock — handle them separately so an `if __name__ ==
        # "__main__":` style script also gets covered.
        new_body = self._maybe_insert_into_top_level(list(updated_node.body))
        if new_body is None:
            return updated_node
        return updated_node.with_changes(body=tuple(new_body))

    def _maybe_insert_after_assignment(self, block: cst.IndentedBlock) -> cst.IndentedBlock:
        body = list(block.body)
        new_body, changed = self._insert(body)
        if not changed:
            return block
        return block.with_changes(body=tuple(new_body))

    def _maybe_insert_into_top_level(
        self, body: list[cst.BaseStatement]
    ) -> list[cst.BaseStatement] | None:
        new_body, changed = self._insert(body)
        if not changed:
            return None
        return new_body

    def _insert(self, body: list[cst.BaseStatement]) -> tuple[list[cst.BaseStatement], bool]:
        if self.applied:
            return body, False
        new_body: list[cst.BaseStatement] = []
        for i, stmt in enumerate(body):
            new_body.append(stmt)
            if self.applied:
                continue  # only wrap the first construction site
            if not _is_model_assignment(stmt, self.model_name):
                continue
            if i + 1 < len(body) and _is_apply_call(body[i + 1], self.model_name):
                continue  # already applied
            apply_stmt = cst.parse_statement(f"{self.model_name}.apply({_INIT_LAMBDA})")
            new_body.append(apply_stmt)
            self.applied = True
        return new_body, self.applied


class KaimingInit(FixTemplate):
    name: ClassVar[str] = "KaimingInit"

    def __init__(self, model_name: str = "model") -> None:
        self.model_name = model_name

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
        inserter = _Inserter(self.model_name)
        new_tree = tree.visit(inserter)
        if not inserter.applied:
            return FixResult(
                applied=False,
                template=self.name,
                old_source=source,
                reason=f"no `{self.model_name} = SomeClass(...)` assignment found",
            )
        return FixResult(
            applied=True,
            template=self.name,
            old_source=source,
            new_source=new_tree.code,
        )
