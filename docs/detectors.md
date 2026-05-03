# Detectors & fix templates

v1 ships eight stream-processing detectors and eight libcst-backed
fix templates. Each detector is server-side: it consumes the broker's
`kind: "metric"` events and emits `kind: "anomaly"` when it fires.
Each fix template is a `libcst.CSTTransformer` that returns a
`FixResult` (with a unified diff) and is **idempotent on its own
output**.

## Detectors

| slug | signal | severity | persistence | demo |
|------|--------|----------|-------------|------|
| `exploding_grad` | global grad-L2 > N× rolling mean (or NaN/Inf loss) | critical | streak ≥ 3 | ⭐⭐⭐⭐⭐ |
| `lr_too_high` | Δloss sign-flip rate > 0.6 over 50 steps | warning | cooldown 50 | ⭐⭐⭐⭐⭐ |
| `dead_relu` | worst-module dead_fraction > 0.5 | warning | streak ≥ 5 | ⭐⭐⭐⭐ |
| `missing_normalize` | first `input_stats` outside [-3, 3] / max > 50 / unsigned-range | warning | single-shot | ⭐⭐⭐⭐ |
| `model_eval_not_called` | val event with `model_training=True` | warning | single-shot | ⭐⭐⭐⭐ |
| `bad_init` | step-0 CE > 2× log(num_classes) | warning | single-shot | ⭐⭐⭐ |
| `vanishing_grad` | min(per_layer_grad_l2) < 1e-7 AND max/min ratio > 100 | warning | streak ≥ 3 | ⭐⭐⭐⭐ |
| `dataloader_bottleneck` | batch_wait_ms / step_time_ms > threshold | warning | streak ≥ 5 | ⭐⭐⭐ |

Robustness contract: every `observe()` returns `None` for malformed
events rather than raising. Detectors never publish — that's the
runner's job.

## Fix templates

| slug | what it inserts | idempotency |
|------|-----------------|-------------|
| `AddGradientClipping` | `clip_grad_norm_(model.parameters(), max_norm=N)` between `loss.backward()` and `optimizer.step()` | skips if a clip call already sits between the two |
| `LowerLearningRate` | replaces the `lr=…` literal in the optimizer constructor | replaces the value, doesn't duplicate |
| `ReplaceActivation` | `nn.ReLU()` → `nn.LeakyReLU(0.01)` (or any pair) | guarded by class match |
| `AddNormalize` | appends `transforms.Normalize(...)` to a `transforms.Compose([...])` list | skips if Normalize already in the list |
| `InsertEvalMode` | wraps `def validate(...)` body with `model.eval()` … `model.train()` | skips if first stmt is already `<m>.eval()` |
| `KaimingInit` | `<model>.apply(lambda _m: ...kaiming_normal_...)` after first construction | applies to the first construction site only |
| `DataLoaderWorkers` | adds/replaces `num_workers=4`, `pin_memory=True`, `persistent_workers=True` on every `DataLoader(...)` | replaces existing values; idempotent on its own output |
| `AddLRScheduler` | `CosineAnnealingLR(optimizer, T_max=N)` after optimizer + `scheduler.step()` after every `optimizer.step()` | skips if a scheduler already follows the optimizer |

Every transformer is parse-error tolerant: malformed source returns
`FixResult(applied=False, reason="parse error: …")` rather than
raising. The diff property uses `difflib.unified_diff` so the output
applies cleanly via the bench grader's `apply_diff()`.

## Adding a detector

1. Subclass `lensml.detectors.base.Detector` with `name = "my_slug"`.
2. Implement `observe(event) -> Anomaly | None`. Be robust to missing
   keys.
3. Register it in `lensml/server/app.py::_default_detectors` (or
   pass via the `detectors=` kwarg of `create_app`).
4. Add a unit test under `tests/unit/test_detectors_my_slug.py`.
5. Add a tiny fixture under `lensml/bench/fixtures/tiny/my_slug/0/`
   if the pathology has a tractable synthetic manifestation.
