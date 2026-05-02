# Changelog

All notable changes to lensml will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Phase 1 — Foundation (Week 1)

End-of-Week-1 state: trainer publishes telemetry → FastAPI server bridges
to dashboard via SSE → first detector fires server-side → dashboard
renders anomaly. Architecture is end-to-end validated.

#### Added

- `lensml.core.hooks` — `TelemetryBuffer` + `install_capture_hooks` (hot-path
  zero-cost forward hooks on activation modules).
- `lensml.core.telemetry` — `_strided_sample`, `activation_stats`,
  `global_grad_l2`, `per_layer_grad_l2` (cold-path summary stats; runs at
  plan §3.1's 10–50 Hz tick rate).
- `lensml.core.transport` — `ZMQPublisher`, `ZMQSubscriber`, `JSONLPublisher`,
  `make_publisher()` factory. Default endpoint `tcp://127.0.0.1:5555`.
- `lensml.server.store` — `EventStore` with the plan §3.3 SQLite schema
  (runs / events / decisions tables, WAL mode).
- `lensml.server.sse` — `EventBroker` (async pub/sub with bounded
  per-subscriber queues) + `format_sse` helper.
- `lensml.server.app` — FastAPI app factory with SSE endpoints, REST
  endpoints, ZMQ-SUB → broker bridge, and CORS middleware for the
  dashboard.
- `lensml.integrations.pytorch` — `lensml.watch(model, optimizer, ...)`
  one-line user API + `WatchHandle` (context manager, `unwatch()`,
  `handle.log(**fields)` for user metrics).
- `lensml.detectors.base` — `Detector` ABC + `Anomaly` dataclass.
- `lensml.detectors.exploding_grad` — first real detector (rolling
  baseline, 10× spike threshold, 3-step persistence, NaN/Inf shortcuts).
- `lensml.detectors.runner` — `DetectorRunner` async task that subscribes
  to the broker, runs all configured detectors on each metric event,
  persists + re-publishes anomalies.
- `lensml.cli dashboard` — starts the FastAPI server via `uvicorn` (string
  factory, supports `--reload`, `--db`, `--zmq`).
- `dashboard/` — Next.js 15.5 + React 19 + Tailwind 4 + uPlot + TypeScript.
  Three-pane layout: live loss chart (uPlot via `useSSE`), agent reasoning
  (renders anomaly events with severity-coloured badge + evidence dl),
  diff preview (placeholder for step 2.5).
- 113 unit + integration tests, all green on CI (Python 3.11/3.12 ×
  Linux/macOS) + dashboard build job.

#### Phase 0 (pre-build de-risking spikes)

- Hooks overhead spike (CPU +1.73%, GPU +0.18 ms/step).
- LiteLLM cross-provider tool-call portability (5/5 providers PASS).
- Subprocess restart spike (12/12 byte-identical losses on CPU and GPU).

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
