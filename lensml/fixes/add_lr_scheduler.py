"""AddLRScheduler — fix for missing learning-rate schedule.

Inserts a ``CosineAnnealingLR`` scheduler immediately after the first
optimizer construction site, and a ``<scheduler>.step()`` call after
each ``<optimizer>.step()`` in the same enclosing block.

Pattern matched (the assignment that triggers insertion):

    <opt_name> = torch.optim.<Cls>(...)

After:

    <opt_name> = torch.optim.<Cls>(...)
    <sched_name> = torch.optim.lr_scheduler.CosineAnnealingLR(<opt_name>, T_max=<T>)

And every ``<opt_name>.step()`` is followed by ``<sched_name>.step()``
(in the same block, immediately after).

Idempotent: skips if a ``<sched_name> = ...`` assignment already
follows the optimizer construction.

Plan §3.2 (template list).
"""

from __future__ import annotations

from typing import ClassVar

import libcst as cst
import libcst.matchers as m

from lensml.fixes.base import FixResult, FixTemplate

DEFAULT_T_MAX = 50
DEFAULT_SCHED_NAME = "scheduler"


def _is_optimizer_assignment(stmt: cst.BaseStatement, opt_name: str) -> bool:
    """`<opt_name> = torch.optim.<Cls>(...)` (or any callable on the RHS)."""
    return m.matches(
        stmt,
        m.SimpleStatementLine(
            body=[
                m.Assign(
                    targets=[m.AssignTarget(target=m.Name(opt_name))],
                    value=m.Call(),
                )
            ]
        ),
    )


def _is_scheduler_assignment(stmt: cst.BaseStatement, sched_name: str) -> bool:
    return m.matches(
        stmt,
        m.SimpleStatementLine(
            body=[
                m.Assign(
                    targets=[m.AssignTarget(target=m.Name(sched_name))],
                    value=m.Call(),
                )
            ]
        ),
    )


def _is_step_call(stmt: cst.BaseStatement, name: str) -> bool:
    return m.matches(
        stmt,
        m.SimpleStatementLine(
            body=[
                m.Expr(
                    value=m.Call(
                        func=m.Attribute(value=m.Name(name), attr=m.Name("step")),
                    )
                )
            ]
        ),
    )


class _Inserter(cst.CSTTransformer):
    def __init__(self, opt_name: str, sched_name: str, t_max: int) -> None:
        super().__init__()
        self.opt_name = opt_name
        self.sched_name = sched_name
        self.t_max = t_max
        self.applied_assignment = False
        self.applied_step = False

    @property
    def applied(self) -> bool:
        return self.applied_assignment or self.applied_step

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        new_body = self._insert_assignment(list(updated_node.body))
        new_body = self._insert_step_calls(new_body)
        if not self.applied:
            return updated_node
        return updated_node.with_changes(body=tuple(new_body))

    def leave_IndentedBlock(
        self,
        original_node: cst.IndentedBlock,
        updated_node: cst.IndentedBlock,
    ) -> cst.IndentedBlock:
        body = list(updated_node.body)
        body = self._insert_assignment(body)
        body = self._insert_step_calls(body)
        # Returning a new block with the same content is fine; libcst
        # treats it as a no-op.
        return updated_node.with_changes(body=tuple(body))

    def _insert_assignment(self, body: list[cst.BaseStatement]) -> list[cst.BaseStatement]:
        if self.applied_assignment:
            return body
        new_body: list[cst.BaseStatement] = []
        for i, stmt in enumerate(body):
            new_body.append(stmt)
            if not _is_optimizer_assignment(stmt, self.opt_name):
                continue
            # Skip if next stmt is already <sched_name> = <Call>
            if i + 1 < len(body) and _is_scheduler_assignment(body[i + 1], self.sched_name):
                continue
            sched_stmt = cst.parse_statement(
                f"{self.sched_name} = "
                f"torch.optim.lr_scheduler.CosineAnnealingLR("
                f"{self.opt_name}, T_max={self.t_max})"
            )
            new_body.append(sched_stmt)
            self.applied_assignment = True
        return new_body

    def _insert_step_calls(self, body: list[cst.BaseStatement]) -> list[cst.BaseStatement]:
        new_body: list[cst.BaseStatement] = []
        for i, stmt in enumerate(body):
            new_body.append(stmt)
            if not _is_step_call(stmt, self.opt_name):
                continue
            # Skip if next stmt is already <sched_name>.step()
            if i + 1 < len(body) and _is_step_call(body[i + 1], self.sched_name):
                continue
            step_stmt = cst.parse_statement(f"{self.sched_name}.step()")
            new_body.append(step_stmt)
            self.applied_step = True
        return new_body


class AddLRScheduler(FixTemplate):
    name: ClassVar[str] = "AddLRScheduler"

    def __init__(
        self,
        optimizer_name: str = "optimizer",
        scheduler_name: str = DEFAULT_SCHED_NAME,
        t_max: int = DEFAULT_T_MAX,
    ) -> None:
        self.optimizer_name = optimizer_name
        self.scheduler_name = scheduler_name
        self.t_max = t_max

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
        inserter = _Inserter(self.optimizer_name, self.scheduler_name, self.t_max)
        new_tree = tree.visit(inserter)
        if not inserter.applied:
            return FixResult(
                applied=False,
                template=self.name,
                old_source=source,
                reason=(
                    f"no `{self.optimizer_name} = ...` assignment found "
                    "(or scheduler already wired)"
                ),
            )
        return FixResult(
            applied=True,
            template=self.name,
            old_source=source,
            new_source=new_tree.code,
        )
