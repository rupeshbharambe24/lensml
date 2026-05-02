"""ReplaceActivation — fix for dead-ReLU and vanishing-grad (plan §4 rows 3, 8).

Replaces calls of the form ``nn.<old>()`` (or ``<old>()`` after a
``from torch.nn import ...``) with ``nn.<new>(<new_args>)``. The default
mapping handles the dead-ReLU case (``nn.ReLU()`` → ``nn.LeakyReLU(0.01)``);
the vanishing-grad case is supported via init params (``Sigmoid`` → ``GELU``).
"""

from __future__ import annotations

from typing import ClassVar

import libcst as cst
import libcst.matchers as m

from lensml.fixes.base import FixResult, FixTemplate


class _ActivationReplacer(cst.CSTTransformer):
    def __init__(self, old: str, new: str, new_args: str) -> None:
        super().__init__()
        self.old = old
        self.new = new
        self.new_args = new_args  # source text for the new call's argument list
        self.applied = False

    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.BaseExpression:
        if not self._matches_old(updated_node):
            return updated_node
        new_call = self._build_new_call(updated_node)
        self.applied = True
        return new_call

    # ---- helpers ----

    def _matches_old(self, call: cst.Call) -> bool:
        # `nn.<old>(...)` or `<old>(...)`
        return m.matches(call.func, m.Attribute(attr=m.Name(self.old))) or m.matches(
            call.func, m.Name(self.old)
        )

    def _build_new_call(self, original: cst.Call) -> cst.Call:
        if isinstance(original.func, cst.Attribute):
            new_func: cst.BaseExpression = original.func.with_changes(attr=cst.Name(self.new))
        else:
            new_func = cst.Name(self.new)

        # Parse the new argument list from the textual form.
        if self.new_args.strip():
            tmp = cst.parse_expression(f"f({self.new_args})")
            assert isinstance(tmp, cst.Call)
            new_args = tmp.args
        else:
            new_args = ()

        return original.with_changes(func=new_func, args=new_args)


class ReplaceActivation(FixTemplate):
    name: ClassVar[str] = "ReplaceActivation"

    def __init__(
        self,
        old: str = "ReLU",
        new: str = "LeakyReLU",
        new_args: str = "0.01",
    ) -> None:
        self.old = old
        self.new = new
        self.new_args = new_args

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
        replacer = _ActivationReplacer(self.old, self.new, self.new_args)
        new_tree = tree.visit(replacer)
        if not replacer.applied:
            return FixResult(
                applied=False,
                template=self.name,
                old_source=source,
                reason=f"no `{self.old}` activation calls found",
            )
        return FixResult(
            applied=True,
            template=self.name,
            old_source=source,
            new_source=new_tree.code,
        )
