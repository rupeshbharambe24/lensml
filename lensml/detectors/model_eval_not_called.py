"""Detector #6 — `model.eval()` not called at validation time.

Signal: at val time, walk modules; if any BN/Dropout has `module.training=True`,
        flag.
Fix:    `insert_eval_mode.InsertEvalMode` — wrap val loop with
        model.eval() / model.train().
Demo:   ⭐⭐⭐⭐ val acc doubles. (Plan §4 row 6.)
"""

from __future__ import annotations

# Implemented Day 17.
