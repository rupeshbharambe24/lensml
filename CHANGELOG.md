# Changelog

All notable changes to lensml will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

End-of-Phase-5 state: every numbered step in `lensml-implementation-sequence.md`
is shipped or marked as user-only manual submission work. **563/563 tests pass**
on CI (Python 3.11/3.12 × Ubuntu/macOS), the dashboard production-builds, mkdocs
strict-builds, the Docker fresh-install smoke runs green.

### Phase 0 — Pre-build de-risking spikes

- Hooks overhead spike (CPU +1.73%, GPU +0.18 ms/step).
- LiteLLM cross-provider tool-call portability (5/5 providers PASS).
- Subprocess restart spike (12/12 byte-identical losses on CPU and GPU).

### Phase 1 — Foundation (Week 1)

Trainer telemetry → FastAPI server → SSE dashboard → first detector,
end-to-end validated.

- `lensml.core.{hooks,telemetry,transport}` — capture hooks, summary
  stats, ZMQ + JSONL publishers with `make_publisher()` factory.
- `lensml.server.{store,sse,app}` — SQLite event log (plan §3.3 schema,
  WAL mode), `EventBroker` async pub/sub with bounded per-subscriber
  queues, FastAPI app factory with SSE + REST + CORS.
- `lensml.integrations.pytorch` — `lensml.watch(model, optimizer, ...)`
  one-line API + `WatchHandle` (context manager, `unwatch()`,
  `handle.log(**fields)`).
- `lensml.detectors.{base,exploding_grad,runner}` — `Detector` ABC +
  `Anomaly` dataclass, first real detector (10× rolling-mean spike,
  3-step persistence, NaN/Inf shortcuts), async runner that subscribes
  to the broker and persists + re-publishes anomalies.
- `lensml.cli dashboard` — `uvicorn` string factory, `--reload`, `--db`,
  `--zmq`.
- `dashboard/` — Next.js 15.5 + React 19 + Tailwind 4 + uPlot. Three-pane
  layout: live loss chart, agent reasoning, diff preview.

### Phase 2 — Magic moment (Week 2)

`detect → diagnose → diff → y → patch committed → checkpoint → restart →
resume`, on raw PyTorch.

- `lensml.llm.{routing,client}` — `LensmlLLM` with LiteLLM fallback
  chains; 4 task categories, 5 providers (Groq, Cerebras, Gemini,
  OpenRouter, Mistral) on free tiers.
- `lensml.agent.{tools,memory,confidence,loop}` — 12 tools with Pydantic
  schemas, `DiagnosisState` + sliding-window `History`, N-way
  self-consistency consensus, ~120-LOC ReAct loop with 8-cycle cap.
- `lensml.fixes/{base,grad_clip,lower_lr,replace_activation,add_normalize}.py`
  — `FixTemplate` ABC + 4 idempotent libcst transformers + `FIX_REGISTRY`.
- Approval flow + git: `decisions.compute_diff_preview` /
  `apply_decision`, `GitRepo` subprocess wrapper (session branches,
  pinned identity, `reset_hard` undo), 4 REST endpoints
  (list/get/approve/reject/undo), dashboard `DiffPreview` (polling,
  color-coded diff, y/n/u keyboard shortcuts).
- `lensml.server.checkpoint` + `lensml.core.signals` — full plan §3.3
  payload (model + optimizer + scheduler + RNG + sampler position),
  cross-platform stop signal (SIGUSR1 Unix / STOP-flag file Windows),
  `setup_determinism` recipe, `EXIT_CHECKPOINTED=99` sentinel.
- `lensml.server.supervisor` — `request_stop`, `is_alive`,
  `read_pid_file`, `spawn_resumed_trainer`.
- `lensml.watch()` integration: `WatchHandle` writes `trainer.pid`,
  installs `CheckpointSignal`, checks `should_stop` every step,
  `LENSML_RESUME` resume on entry, public `handle.checkpoint()`.

### Phase 3 — Breadth + polish (Week 3)

Three integrations, all 8 detectors, all 8 fix templates, dashboard
production-grade, replay mode, privacy scaffold.

- `LensmlCallback` (PyTorch Lightning) — installs hooks + signal in
  `on_train_start`, telemetry + stop check in `on_before_optimizer_step`,
  injects `lensml` metadata into Lightning checkpoints.
- `LensmlTrainerCallback` (HuggingFace) — read-only callback contract:
  stop signal sets `control.should_save = True` +
  `control.should_training_stop = True`, then `on_train_end` exits with
  `EXIT_CHECKPOINTED`.
- 7 new detectors: `dead_relu`, `lr_too_high`, `missing_normalize`,
  `model_eval_not_called`, `dataloader_bottleneck`, `bad_init`,
  `vanishing_grad` (single-shot or persisted, all robust to malformed
  events).
- 4 new fix templates: `InsertEvalMode`, `KaimingInit`,
  `DataLoaderWorkers`, `AddLRScheduler`. All 8 templates idempotent +
  parse-error tolerant + registered.
- Dashboard polish (step 3.5): vertical intervention markers on the
  loss chart (uPlot draw hook), pop-in animation on new anomalies,
  status pill with step counter, `cmdk` command palette (⌘K), `?`
  keyboard help overlay, EN/JA i18n with localStorage persistence,
  EN/JA toggle button.
- Replay mode: `Player` (sync + async, ts-driven + fixed-delay pacing),
  `Recorder` (broker → JSONL), 3 deterministic scenarios
  (`exploding_grad`, `dead_relu`, `missing_normalize`) round-trip-
  verified through their real detectors.
- New CLI subcommands: `lensml replay [--scenario | JSONL]`,
  `lensml record OUT --scenario NAME`.
- Privacy mode (Plan-C scaffold): `PrivacyConfig` + `canned_diagnosis`
  pure function covering all 8 v1 detectors, REST endpoints
  (`GET/POST /api/privacy`, `POST /api/privacy/canned`), dashboard
  toggle pill + `p` shortcut + palette entry, "Run local diagnosis"
  button on the AgentReasoning card.

### Phase 4 — Bench + video + landing (Week 4)

- `lensml.bench.{fixtures,grader}` — `Fixture` loader with meta-JSON
  validation (12 pathology classes, 4 envs, 3 difficulties), subprocess
  grader (plan §6.1 three-check protocol via final-line JSON status),
  `apply_diff` minimal hunk walker.
- 8 packaged Tiny fixtures (one per v1 detector class with a tractable
  manifestation, plus `lr_schedule_mismatch` stretch). Each: `bug.py`
  + difflib-generated `fix.diff` + `grade.py` shim + `meta.json`.
  `dataloader_bottleneck` documented as small/medium-only.
- `lensml.bench.runner` — `BenchRunner.run_one` + `run_all` with
  env/pathology filters and JSON resume cache.
- `lensml.bench.baseline` — LLM-only baseline arm (plan §6.3
  credibility lynchpin). `parse_diagnosis_class` (synonyms → 12
  canonical slugs), `parse_diff_from_response` (fenced-block extractor),
  `BaselineRunner` with mockable `complete(prompt) -> str` callable.
- `lensml.bench.report` — markdown + JSON renderer with confusion
  matrix per arm (diagonal bolded), latency histogram (shared bucket
  edges), `$0` cost breakdown, side-by-side comparison, "what doesn't
  work yet" section.
- New CLI subgroup: `lensml bench {list,run,baseline,report}`.
- `demo/{install,watch,intervene}.tape` — VHS scripts for the README
  hero GIFs. `demo/scripts/demo_train.py` drives the watch GIF.
  `demo/RECORDING.md` is the 90-second video playbook.
- `landing/index.html` — single-file Vercel free-tier landing page
  (~6 KB gzipped, dashboard-matched palette).
- `mkdocs.yml` + `docs/{index,quickstart,detectors,bench,roadmap}.md`
  — Material theme, dark/light cyan accent. `mkdocs build --strict`
  clean. `.github/workflows/docs.yml` auto-deploys to gh-pages on main.

### Phase 5 — Submission (Week 5)

Infrastructure for the manual submit-day actions.

- `Dockerfile` + `.dockerignore` — fresh-environment install image:
  `docker build -t lensml-smoke . && docker run --rm lensml-smoke`.
- `scripts/smoke_install.sh` — validates every public submodule,
  all 8 fix templates, all 8 packaged fixtures, all 3 replay scenarios.
  CI runs it on every push.
- `submission/` — `pitch.md` (paste-ready descriptions), `links.md`
  (single source of truth), `qa.md` (pre-baked answers to anticipated
  judge questions), `submission_checklist.md` (Sunday T-5h timeline),
  `launch.md` (drafts: tweet thread, LinkedIn, Show HN — HN held
  until after R2).

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
