# Roadmap

The 30-day FAR AWAY hackathon build (May–Jun 2026) ships v0.1.
The implementation sequence file
([here](https://github.com/rupeshbharambe24/lensml/blob/main/lensml-implementation-sequence.md))
records exactly what shipped on which day.

## v0.1 (FAR AWAY Round 1, June 7 2026)

- ✅ `lensml.watch()` for raw PyTorch with checkpoint resume + signal handling
- ✅ Lightning callback (`LensmlCallback`)
- ✅ HF Trainer callback (`LensmlTrainerCallback`)
- ✅ All 8 v1 detectors
- ✅ All 8 libcst fix templates
- ✅ FastAPI server + SQLite event log + ZMQ bridge
- ✅ Next.js dashboard with command palette + EN/JA i18n + privacy toggle
- ✅ Replay mode + 3 deterministic demo scenarios
- ✅ `lensml-bench` infrastructure + 8 tiny fixtures + LLM-only baseline arm
- ⏳ Final benchmark report (numbers populate from external Colab/Kaggle runs)
- ⏳ 90-second Round 1 video (recording infra in `demo/`; manual recording step)

## v0.2 (Q3 2026)

- Distributed-training support (DDP, FSDP, accelerate)
- Medium and Large bench envs (DistilBERT/SST-2, GPT-2/WikiText-2)
- Real Ollama integration for privacy mode (today's scaffolding is
  Plan-C: toggle + canned interaction; v0.2 wires the local model)
- 4 stretch detectors fully shipped (label_leakage, train_val_contamination,
  mixed_precision_nan, lr_schedule_mismatch — `lr_schedule_mismatch` already
  has a tiny fixture)
- Custom-detector API: `lensml.detectors.register(my_detector)`
- Replay → mp4 export (today: live SSE only)

## v0.3

- Fine-tuned supervisor models for the diagnosis path (today: prompt-only)
- Built-in pre-baked fixes for top-20 PyTorch antipatterns
- Notebook (Jupyter) integration

## v1.0+ (enterprise)

Code under `/ee/` lives behind a separate license (see `LICENSE-EE`).
Planned: SSO, team dashboards, audit logs, on-prem deployment guide.
The OSS core stays Apache-2.0 forever.
