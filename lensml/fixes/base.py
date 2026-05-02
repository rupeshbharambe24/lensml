"""Base class for libcst-backed fix templates.

Plan refs:
    §2  — libcst preserves comments + whitespace on round-trip; CSTTransformer
          + matchers are the right APIs. Edge cases exist for very new syntax
          but our target patterns (function calls, kwargs, simple statements)
          are stable.
    §4  — fix templates are 1:1 with detector pathology classes.
    §3.2 — modules under `lensml/fixes/`.
    §11 risk #2 — libcst f-string / type-hint edge cases break a fix template
          on a real fixture. Mitigation: each transformer has a unit test on
          a known input (this step), plus fixtures that exercise modern
          syntax (added Day 11+).

Contract:
    - `apply(source)` NEVER raises. On parse error or pattern miss, returns
      a FixResult with `applied=False` + a `reason` string.
    - The transformer is a pure function of source text → new source text.
      It does NOT touch the filesystem or git — that's step 2.5's job.
"""

from __future__ import annotations

import difflib
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class FixResult:
    """Outcome of attempting to apply a fix template to a source string."""

    applied: bool
    template: str
    old_source: str
    new_source: str = ""
    reason: str | None = None

    @property
    def diff(self) -> str:
        """Unified diff between old and new source. Empty if not applied."""
        if not self.applied:
            return ""
        return "".join(
            difflib.unified_diff(
                self.old_source.splitlines(keepends=True),
                self.new_source.splitlines(keepends=True),
                fromfile="before.py",
                tofile="after.py",
            )
        )


class FixTemplate(ABC):
    """Abstract base class for all v1 fix templates."""

    name: str = "base"

    @abstractmethod
    def apply(self, source: str) -> FixResult:
        """Apply the transformation to `source`. Always returns a FixResult."""
