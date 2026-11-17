# lensml

> The AI agent that supervises your ML training runs.

lensml is a Python package + local web dashboard that watches your
PyTorch training run in real time, detects pathologies as they
emerge (exploding gradients, dead ReLUs, missing input
normalization, …), proposes a libcst-level patch, commits it to a
session branch on your approval, and resumes from the last
checkpoint with the fixed code.

The pitch in one sentence: **"integration, not invention"** — every
piece of the pipeline already exists; lensml is the open-source
agent that wires them together.

## Try it

```bash
pip install "git+https://github.com/rupeshbharambe24/lensml.git"   # not on PyPI yet
lensml dashboard &        # http://localhost:8000
# in your training script:
import lensml
lensml.watch(model, optimizer)
```

## Why lensml?

| | W&B / MLflow | TrainCheck | AutoTrainer | Cursor / Aider | **lensml** |
|---|---|---|---|---|---|
| Detect | passive | yes | yes | — | **yes** |
| Diagnose | — | — | rule-based | — | **LLM agent** |
| Patch source | — | — | in-memory (Keras) | offline | **libcst** |
| Resume from ckpt | — | — | — | — | **yes** |

## Status

Pre-release alpha — built solo on free tiers for the FAR AWAY
hackathon. See the
[implementation sequence](https://github.com/rupeshbharambe24/lensml/blob/main/lensml-implementation-sequence.md)
for the day-by-day build log.

## Where to next

- [Architecture](DESIGN.md) — the public design doc.
- [Quickstart](quickstart.md) — `lensml.watch()` end-to-end on a tiny CIFAR run.
- [Detectors & fixes](detectors.md) — the v1 pathology catalogue and the libcst templates that fix them.
- [Bench](bench.md) — `lensml-bench` philosophy + how to add a fixture.
- [Roadmap](roadmap.md) — what's next.
