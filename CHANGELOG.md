# Changelog

All notable changes to lensml will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project scaffold (file/folder structure per `lensml-plan.md` §3.2).

## [0.1.0] - 2026-06-07 (planned)

Round-1 hackathon submission build. Target feature set:

- `lensml.watch(model, optimizer, dataloader)` one-line API
- Three integrations: raw PyTorch, PyTorch Lightning, HuggingFace Trainer
- Eight detectors: exploding gradients, LR too high, dead ReLU, missing input
  normalization, dataloader bottleneck, `model.eval()` not called, bad init,
  vanishing gradients
- Eight libcst fix templates matched to the detectors above
- Single-agent ReAct loop with N=3 self-consistency confidence calibration
- LiteLLM-backed multi-provider routing (Groq / Cerebras / Gemini / OpenRouter / Mistral)
- FastAPI server + SQLite event log + git branch management
- Checkpoint-aware subprocess restart (model + optimizer + scheduler + RNG + sampler)
- Next.js 15 dashboard with uPlot live charts and react-diff-viewer-continued
- `lensml replay <jsonl>` deterministic playback
- Local privacy mode (scaffolded; cloud is the demo path)
- `lensml-bench`: 48 fixtures across 12 pathology classes × 4 host environments
