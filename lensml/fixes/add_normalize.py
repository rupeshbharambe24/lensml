"""AddNormalize — fix for missing input normalization (plan §4 row 4).

Inserts a ``transforms.Normalize(mean=..., std=...)`` element at the end of
the list passed to any ``transforms.Compose([...])`` call. Defaults match
ImageNet stats per plan §4 row 4.

Skips the call if Normalize is already present (idempotent).
"""

from __future__ import annotations

from typing import ClassVar

import libcst as cst
import libcst.matchers as m

from lensml.fixes.base import FixResult, FixTemplate

_DEFAULT_MEAN = (0.485, 0.456, 0.406)
_DEFAULT_STD = (0.229, 0.224, 0.225)


def _is_compose_call(node: cst.Call) -> bool:
    return m.matches(node.func, m.Attribute(attr=m.Name("Compose"))) or m.matches(
        node.func, m.Name("Compose")
    )


def _has_normalize_element(elements: list[cst.BaseElement]) -> bool:
    for elem in elements:
        if not isinstance(elem, cst.Element):
            continue
        v = elem.value
        if isinstance(v, cst.Call) and (
            m.matches(v.func, m.Attribute(attr=m.Name("Normalize")))
            or m.matches(v.func, m.Name("Normalize"))
        ):
            return True
    return False


class _NormalizeAdder(cst.CSTTransformer):
    def __init__(self, mean: tuple[float, ...], std: tuple[float, ...]) -> None:
        super().__init__()
        self.mean = mean
        self.std = std
        self.applied = False

    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.Call:
        if not _is_compose_call(updated_node):
            return updated_node
        if not updated_node.args:
            return updated_node

        first_arg = updated_node.args[0].value
        if not isinstance(first_arg, cst.List):
            return updated_node

        elements = list(first_arg.elements)
        if _has_normalize_element(elements):
            return updated_node

        normalize_expr = cst.parse_expression(
            f"transforms.Normalize(mean={list(self.mean)}, std={list(self.std)})"
        )
        # Preserve the existing list's trailing comma style by reusing the
        # last element's comma if present.
        new_elem = cst.Element(value=normalize_expr)
        if elements and not isinstance(elements[-1].comma, cst.MaybeSentinel):
            # Last existing element already has a comma → just append.
            pass
        elements.append(new_elem)

        new_list = first_arg.with_changes(elements=tuple(elements))
        new_first_arg = updated_node.args[0].with_changes(value=new_list)
        new_args = (new_first_arg, *updated_node.args[1:])
        self.applied = True
        return updated_node.with_changes(args=new_args)


class AddNormalize(FixTemplate):
    name: ClassVar[str] = "AddNormalize"

    def __init__(
        self,
        mean: tuple[float, ...] = _DEFAULT_MEAN,
        std: tuple[float, ...] = _DEFAULT_STD,
    ) -> None:
        self.mean = mean
        self.std = std

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
        adder = _NormalizeAdder(self.mean, self.std)
        new_tree = tree.visit(adder)
        if not adder.applied:
            return FixResult(
                applied=False,
                template=self.name,
                old_source=source,
                reason="no `transforms.Compose([...])` call found (or Normalize already present)",
            )
        return FixResult(
            applied=True,
            template=self.name,
            old_source=source,
            new_source=new_tree.code,
        )
