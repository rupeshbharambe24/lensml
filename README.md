# lensml

> The AI agent that supervises your ML training runs.

<!-- Hero GIF placeholder — record on Day 27 (see lensml-plan.md §8) -->

[![PyPI](https://img.shields.io/pypi/v/lensml.svg)](https://pypi.org/project/lensml/)
[![Python](https://img.shields.io/pypi/pyversions/lensml.svg)](https://pypi.org/project/lensml/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![CI](https://github.com/rupeshbharambe24/lensml/actions/workflows/ci.yml/badge.svg)](https://github.com/rupeshbharambe24/lensml/actions/workflows/ci.yml)

## What it does

lensml watches your PyTorch / PyTorch Lightning / HuggingFace Trainer training run in real time. When it detects a pathology (exploding gradients, dead ReLUs, missing normalization, `model.eval()` not called, etc.), it pauses training, diagnoses the cause with an LLM agent, generates a code patch via libcst AST rewrite, and presents a diff in a local browser dashboard. On your approval (one keystroke), it commits the patch to a git branch, checkpoints model + optimizer + RNG + sampler state, and resumes training from the checkpoint with the fixed code.

- Three integrations: raw PyTorch, PyTorch Lightning, HuggingFace Trainer
- Eight v1 pathology classes (see [docs](docs/))
- Free LLM tier routing across Groq / Cerebras / Gemini / OpenRouter via LiteLLM
- Optional local privacy mode via Ollama
- 48-fixture benchmark suite (`lensml-bench`)

## 30-second install

```bash
pip install lensml
echo "import lensml; lensml.watch(model, optimizer, dataloader)" >> train.py
lensml dashboard  # opens browser
```

## Why lensml?

| Tool | Detect | Diagnose | Patch source | Resume from ckpt |
|---|---|---|---|---|
| W&B / MLflow / Neptune | passive metrics | — | — | — |
| TrainCheck (OSDI 2025) | yes | — | — | — |
| AutoTrainer (ICSE 2021) | yes | rule-based | in-memory only (Keras) | — |
| Cursor / Aider | — | — | yes (offline) | — |
| **lensml** | **yes** | **LLM agent** | **yes (libcst)** | **yes** |

## Architecture

See [DESIGN.md](docs/DESIGN.md) for the public design doc. Short version:

```
Training subprocess  ─ZMQ PUB/SUB→  FastAPI server  ─SSE→  Next.js dashboard
(user's train.py +                  (agent loop +          (uPlot charts +
 lensml.watch hooks)                 git + ckpt +           diff viewer +
                     ←─SIGUSR1───   SQLite event log)       y/n/u keys)
```

## Benchmarks

48 fixtures across 12 pathology classes × 4 host environments (MNIST/MLP, CIFAR/ResNet-18, SST-2/DistilBERT, WikiText/GPT-2). Metrics: top-1 / top-3 diagnosis accuracy, pass@1 / pass@3 fix rate, time-to-diagnose, tokens-to-diagnose, cost-to-diagnose.

Results: `lensml/bench/results/` (populated Week 4 of build).

## Roadmap

- **v0.2** HF Trainer auto-restart, distributed training (DDP/FSDP), more pathology classes
- **v0.3** Custom detector API, fine-tuned supervisor models
- **v1.0** Enterprise features in `/ee/` (SSO, team dashboards)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0 for the core (see [LICENSE](LICENSE)). Future enterprise features live in `/ee/` under a separate license (see [LICENSE-EE](LICENSE-EE)).
