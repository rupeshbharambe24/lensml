"""AddGradientClipping — fix for exploding gradients.

Inserts `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
between `loss.backward()` and `optimizer.step()`. Adds the import via
AddImportsVisitor.

Plan §4 row 1. Implemented Day 11.
"""

from __future__ import annotations
