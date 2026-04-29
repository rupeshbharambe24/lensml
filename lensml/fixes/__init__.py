"""lensml.fixes — libcst-backed source transformers.

Per plan §3.2, v1 ships these templates:
    base.py                FixTemplate ABC
    grad_clip.py           AddGradientClipping
    lower_lr.py            LowerLearningRate
    replace_activation.py  ReplaceActivation (ReLU/Sigmoid → LeakyReLU/GELU)
    add_normalize.py       AddNormalize
    insert_eval_mode.py    InsertEvalMode
    kaiming_init.py        KaimingInit
    dataloader_workers.py  DataLoaderWorkers
    add_lr_scheduler.py    AddLRScheduler

Per plan §2: libcst preserves comments and whitespace on round-trip.
CSTTransformer + libcst.matchers is the right API. AddImportsVisitor
handles imports cleanly. Edge cases exist for very new syntax but the
target patterns (function calls, kwargs, simple statements) are stable.
"""
