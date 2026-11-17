# Quickstart

This page walks you from `pip install` to seeing the dashboard
react to a real training run in about five minutes.

## 1. Install

lensml is not on PyPI yet, so install from GitHub:

```bash
pip install "git+https://github.com/rupeshbharambe24/lensml.git"                       # core
pip install "lensml[lightning] @ git+https://github.com/rupeshbharambe24/lensml.git"    # for Lightning callback
pip install "lensml[transformers] @ git+https://github.com/rupeshbharambe24/lensml.git" # for HF Trainer callback
```

## 2. Start the dashboard

```bash
lensml dashboard --db lensml.sqlite
```

That's a FastAPI server on `:8000`. The Next.js dashboard lives in
`dashboard/` of the repo — `cd dashboard && pnpm install && pnpm
dev` to bring it up on `:3000`. The dashboard talks directly to
`:8000` over CORS (SSE doesn't survive Next's dev proxy).

## 3. Add `lensml.watch()` to your training loop

```python
import lensml
import torch.optim as optim

model = MyNet()
optimizer = optim.SGD(model.parameters(), lr=0.05)

handle = lensml.watch(model, optimizer, run_id="my-run")
try:
    for step in range(N):
        loss = train_step(model, optimizer)
        handle.log(loss=loss.item())     # optional — stream user metrics too
finally:
    handle.unwatch()
```

That's it. Hooks are installed on activation modules, `optimizer.step`
is wrapped to capture pre-step gradients, and a 50ms-cadence
telemetry stream goes to `:8000`. The dashboard's loss curve, grad-L2
trace, and agent reasoning pane update live.

## 4. Lightning

```python
from lensml.integrations.lightning import LensmlCallback, lensml_resume_ckpt

trainer = pl.Trainer(callbacks=[LensmlCallback()], ...)
trainer.fit(model, datamodule, ckpt_path=lensml_resume_ckpt())
```

`lensml_resume_ckpt()` reads `LENSML_RESUME` so the supervisor's
restart hand-off works automatically.

## 5. HuggingFace Trainer

```python
from lensml.integrations.transformers import LensmlTrainerCallback, lensml_resume_ckpt

trainer = Trainer(model=..., args=..., callbacks=[LensmlTrainerCallback()])
trainer.train(resume_from_checkpoint=lensml_resume_ckpt())
```

HF's TrainerCallback is read-only, so lensml signals stop via
`control.should_training_stop = True`. The callback then `sys.exit(99)`
on `on_train_end` so the supervisor sees the sentinel.

## 6. Replay (for offline demos)

```bash
lensml replay --scenario exploding_grad --speed 2
```

Streams a deterministic recorded session through the dashboard. No
GPU, no LLM, no flakiness — perfect for the live demo path.

## 7. Bench

```bash
lensml bench list                          # 8 packaged tiny fixtures
lensml bench run --out bench/results/$(date +%F)/
lensml bench baseline --provider groq --out bench/results/$(date +%F)/
lensml bench report \
    --bench bench/results/$(date +%F)/report.json \
    --baseline bench/results/$(date +%F)/baseline-groq.json \
    --out bench/results/$(date +%F)/
```
