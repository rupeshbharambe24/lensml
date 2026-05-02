"""InsertEvalMode — fix for ``model.eval()`` not called at validation
(plan §4 row 6).

Pattern (heuristic):
    1. Find any function whose name suggests validation
       (``validate``, ``evaluate``, ``val_step``, ``validation_step``,
       ``test_step``, ``run_eval``, …).
    2. Prepend ``<model>.eval()`` as the first statement of the body.
    3. Append ``<model>.train()`` as the last statement of the body.

Idempotent: skips functions whose body already starts with
``<model>.eval()``.

The transformer doesn't try to handle decorated returns, early-return
control flow, or context-manager wrapping — those edges are handled
by the agent suggesting the user manually wrap the val loop. v1
covers the 80% case (a vanilla ``def validate(model, loader): ...``).
"""

from __future__ import annotations

from typing import ClassVar

import libcst as cst
import libcst.matchers as m

from lensml.fixes.base import FixResult, FixTemplate

_VAL_NAME_KEYWORDS = (
    "validate",
    "validation",
    "evaluate",
    "evaluation",
    "val_step",
    "test_step",
    "run_eval",
    "do_eval",
)


def _is_target_function_name(name: str) -> bool:
    n = name.lower()
    return any(kw in n for kw in _VAL_NAME_KEYWORDS)


def _is_eval_call(stmt: cst.BaseStatement, model_name: str) -> bool:
    """True if `stmt` is `<model_name>.eval()`."""
    return m.matches(
        stmt,
        m.SimpleStatementLine(
            body=[
                m.Expr(
                    value=m.Call(
                        func=m.Attribute(value=m.Name(model_name), attr=m.Name("eval")),
                        args=[],
                    )
                )
            ]
        ),
    )


def _is_train_call(stmt: cst.BaseStatement, model_name: str) -> bool:
    return m.matches(
        stmt,
        m.SimpleStatementLine(
            body=[
                m.Expr(
                    value=m.Call(
                        func=m.Attribute(value=m.Name(model_name), attr=m.Name("train")),
                        args=[],
                    )
                )
            ]
        ),
    )


class _Inserter(cst.CSTTransformer):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.model_name = model_name
        self.applied = False

    def leave_FunctionDef(
        self,
        original_node: cst.FunctionDef,
        updated_node: cst.FunctionDef,
    ) -> cst.FunctionDef:
        if not _is_target_function_name(updated_node.name.value):
            return updated_node
        if not isinstance(updated_node.body, cst.IndentedBlock):
            return updated_node
        body = list(updated_node.body.body)
        if not body:
            return updated_node

        first = body[0]
        already_has_eval = _is_eval_call(first, self.model_name)

        new_body: list[cst.BaseStatement] = list(body)
        if not already_has_eval:
            eval_stmt = cst.parse_statement(f"{self.model_name}.eval()")
            new_body.insert(0, eval_stmt)
            self.applied = True

        last = new_body[-1]
        if not _is_train_call(last, self.model_name):
            train_stmt = cst.parse_statement(f"{self.model_name}.train()")
            new_body.append(train_stmt)
            self.applied = True

        if not self.applied:
            return updated_node
        new_block = updated_node.body.with_changes(body=tuple(new_body))
        return updated_node.with_changes(body=new_block)


class InsertEvalMode(FixTemplate):
    name: ClassVar[str] = "InsertEvalMode"

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
                reason="no validation-named function found (or eval/train calls already present)",
            )
        return FixResult(
            applied=True,
            template=self.name,
            old_source=source,
            new_source=new_tree.code,
        )
