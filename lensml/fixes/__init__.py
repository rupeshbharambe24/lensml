"""lensml.fixes — libcst-backed source transformers (plan §3.2).

v1 ships these templates:
    base.py                FixTemplate ABC + FixResult
    grad_clip.py           AddGradientClipping       (plan §4 row 1)
    lower_lr.py            LowerLearningRate          (plan §4 row 2)
    replace_activation.py  ReplaceActivation          (plan §4 rows 3, 8)
    add_normalize.py       AddNormalize               (plan §4 row 4)

The remaining templates listed in plan §3.2 (insert_eval_mode, kaiming_init,
dataloader_workers, add_lr_scheduler) land in step 3.4 — their detector
counterparts are still being built.

Per plan §2: libcst preserves comments and whitespace on round-trip.
CSTTransformer + libcst.matchers is the right API. Edge cases exist for
very new syntax but our target patterns (function calls, kwargs, simple
statements) are stable.
"""

from __future__ import annotations

from lensml.fixes.add_lr_scheduler import AddLRScheduler
from lensml.fixes.add_normalize import AddNormalize
from lensml.fixes.base import FixResult, FixTemplate
from lensml.fixes.dataloader_workers import DataLoaderWorkers
from lensml.fixes.grad_clip import AddGradientClipping
from lensml.fixes.insert_eval_mode import InsertEvalMode
from lensml.fixes.kaiming_init import KaimingInit
from lensml.fixes.lower_lr import LowerLearningRate
from lensml.fixes.replace_activation import ReplaceActivation

# Slug → class registry. Keys MUST match each class's `name` attribute.
FIX_REGISTRY: dict[str, type[FixTemplate]] = {
    AddGradientClipping.name: AddGradientClipping,
    LowerLearningRate.name: LowerLearningRate,
    ReplaceActivation.name: ReplaceActivation,
    AddNormalize.name: AddNormalize,
    InsertEvalMode.name: InsertEvalMode,
    KaimingInit.name: KaimingInit,
    DataLoaderWorkers.name: DataLoaderWorkers,
    AddLRScheduler.name: AddLRScheduler,
}


def build_fix(template: str, params: dict | None = None) -> FixTemplate:
    """Construct a fix template instance by slug.

    Raises ``ValueError`` for unknown slugs or invalid params.
    """
    cls = FIX_REGISTRY.get(template)
    if cls is None:
        raise ValueError(f"unknown fix template: {template!r}")
    try:
        return cls(**(params or {}))
    except TypeError as e:
        raise ValueError(f"invalid params for {template}: {e}") from e


__all__ = [
    "FIX_REGISTRY",
    "AddGradientClipping",
    "AddLRScheduler",
    "AddNormalize",
    "DataLoaderWorkers",
    "FixResult",
    "FixTemplate",
    "InsertEvalMode",
    "KaimingInit",
    "LowerLearningRate",
    "ReplaceActivation",
    "build_fix",
]
