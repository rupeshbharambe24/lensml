"""KaimingInit — fix for bad initialization (loss-at-init too high).

Adds `model.apply(_init)` after model construction, with `_init` doing
Kaiming-normal for ReLU layers.

Plan §4 row 7. Implemented Day 18.
"""

from __future__ import annotations
