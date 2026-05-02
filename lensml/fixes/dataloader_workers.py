"""DataLoaderWorkers — fix for DataLoader bottleneck (plan §4 row 5).

Edits ``DataLoader(...)`` call sites to add or update::

    num_workers=<N>
    pin_memory=True
    persistent_workers=True

Where ``<N>`` defaults to 4 (a safe baseline that gives most of the
throughput win without the memory cost of 8+). If a kwarg is already
present its value is replaced; otherwise it's appended.

Idempotent: applying the fix twice is a no-op.
"""

from __future__ import annotations

from typing import ClassVar

import libcst as cst
import libcst.matchers as m

from lensml.fixes.base import FixResult, FixTemplate

DEFAULT_NUM_WORKERS = 4

_TARGET_KWARGS: tuple[tuple[str, str], ...] = (
    ("num_workers", "{n}"),
    ("pin_memory", "True"),
    ("persistent_workers", "True"),
)


def _is_dataloader_call(node: cst.Call) -> bool:
    return m.matches(node.func, m.Name("DataLoader")) or m.matches(
        node.func, m.Attribute(attr=m.Name("DataLoader"))
    )


class _Editor(cst.CSTTransformer):
    def __init__(self, num_workers: int) -> None:
        super().__init__()
        self.num_workers = num_workers
        self.applied = False

    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.Call:
        if not _is_dataloader_call(updated_node):
            return updated_node

        desired: dict[str, str] = {
            "num_workers": str(self.num_workers),
            "pin_memory": "True",
            "persistent_workers": "True",
        }

        new_args: list[cst.Arg] = []
        seen: set[str] = set()
        any_change = False
        for arg in updated_node.args:
            kw = arg.keyword.value if isinstance(arg.keyword, cst.Name) else None
            if kw is not None and kw in desired:
                seen.add(kw)
                desired_value = cst.parse_expression(desired[kw])
                if cst.Module(body=[]).code_for_node(arg.value) == desired[kw]:
                    new_args.append(arg)
                else:
                    new_args.append(arg.with_changes(value=desired_value))
                    any_change = True
            else:
                new_args.append(arg)

        for kw, val_src in desired.items():
            if kw in seen:
                continue
            new_arg = cst.Arg(
                value=cst.parse_expression(val_src),
                keyword=cst.Name(kw),
                equal=cst.AssignEqual(
                    whitespace_before=cst.SimpleWhitespace(""),
                    whitespace_after=cst.SimpleWhitespace(""),
                ),
            )
            new_args.append(new_arg)
            any_change = True

        # Re-thread trailing-comma / spacing on the previous final positional
        # arg (a missing comma between args is a syntax error).
        if any_change and new_args:
            for idx in range(len(new_args) - 1):
                a = new_args[idx]
                if a.comma is cst.MaybeSentinel.DEFAULT or isinstance(a.comma, cst.MaybeSentinel):
                    new_args[idx] = a.with_changes(
                        comma=cst.Comma(
                            whitespace_after=cst.SimpleWhitespace(" "),
                        )
                    )

        if not any_change:
            return updated_node
        self.applied = True
        return updated_node.with_changes(args=tuple(new_args))


class DataLoaderWorkers(FixTemplate):
    name: ClassVar[str] = "DataLoaderWorkers"

    def __init__(self, num_workers: int = DEFAULT_NUM_WORKERS) -> None:
        self.num_workers = num_workers

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
        editor = _Editor(self.num_workers)
        new_tree = tree.visit(editor)
        if not editor.applied:
            return FixResult(
                applied=False,
                template=self.name,
                old_source=source,
                reason="no `DataLoader(...)` call found (or already configured)",
            )
        return FixResult(
            applied=True,
            template=self.name,
            old_source=source,
            new_source=new_tree.code,
        )
