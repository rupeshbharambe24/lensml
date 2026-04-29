"""InsertEvalMode — fix for `model.eval()` not called at validation.

Inserts `model.eval()` at val-loop entry and `model.train()` after.

Plan §4 row 6. Implemented Day 18.
"""

from __future__ import annotations
