"""Detector #6 — ``model.eval()`` not called at validation time
(plan §4 row 6).

Signal: a validation event arrives with ``model_training=True``, i.e.
the trainer ran a validation pass while the model was still in
training mode. BatchNorm's running statistics and Dropout's masks
behave very differently in train vs eval mode, so doing val in train
mode silently halves accuracy.

Fix:    ``insert_eval_mode.InsertEvalMode`` — wrap val loop with
        ``model.eval() / model.train()``.
Demo:   ⭐⭐⭐⭐ val acc doubles.

Reads ``event["phase"] == "val"`` (or the explicit
``kind: "validation"``) plus ``event["model_training"]`` published by
the integrations when entering a validation pass.

Fires at most once per run (single-shot) — the fix is structural, not
something to re-emit on every val batch.
"""

from __future__ import annotations

from typing import Any, ClassVar

from lensml.detectors.base import Anomaly, Detector


class ModelEvalNotCalledDetector(Detector):
    name: ClassVar[str] = "model_eval_not_called"

    def __init__(self) -> None:
        self._fired = False

    def observe(self, event: dict[str, Any]) -> Anomaly | None:
        if self._fired:
            return None

        kind = event.get("kind")
        is_validation = kind == "validation" or (kind == "metric" and event.get("phase") == "val")
        if not is_validation:
            return None

        # The trainer must opt in by publishing model_training. If not present,
        # the detector has no evidence either way and stays silent.
        training_flag = event.get("model_training")
        if not isinstance(training_flag, bool):
            return None
        if training_flag is False:
            return None

        run_id = str(event.get("run_id", "default"))
        step = int(event.get("step", 0)) if isinstance(event.get("step"), int) else 0
        self._fired = True
        # Optional list of offending submodules from the trainer (BN/Dropout).
        offenders = event.get("training_modules")
        offender_list: list[str] | None = None
        if isinstance(offenders, list):
            offender_list = [str(x) for x in offenders if isinstance(x, str)]

        return Anomaly(
            detector=self.name,
            run_id=run_id,
            step=step,
            signal="val_in_train_mode",
            severity="warning",
            evidence={
                "model_training": True,
                "modules": offender_list,
                "phase": event.get("phase"),
            },
        )
