"""lensml.detectors — pathology detectors.

Each detector consumes a stream of telemetry events and emits a structured
`Anomaly` to the agent. Per plan §4, v1 ships 8 detectors:

    1. exploding_grad        Day 6  — global grad-L2 > 10× rolling mean
    2. lr_too_high           Day 17 — Δloss sign-flip rate > 0.6
    3. dead_relu             Day 17 — forward-hook dead fraction > 0.5
    4. missing_normalize     Day 17 — first-batch image mean ∉ [-3, 3]
    5. dataloader_bottleneck Day 18 — GPU idle > 30% + num_workers=0
    6. model_eval_not_called Day 17 — BN/Dropout `module.training` at val
    7. bad_init              Day 18 — step-0 CE > 2× expected log(K)
    8. vanishing_grad        Day 18 — deepest layer grad < 1e-7

NOTE: plan §3.2's module tree omits the `detectors/` folder, but Days 6/17/18
explicitly reference `lensml.detectors.*`. The folder is included here on
that basis.
"""
