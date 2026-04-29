# lensml — design

This is the public-facing design doc (plan §13, ~900 words). It is the
canonical "what is lensml" reference for users and contributors.

> **Stub.** Final text is taken verbatim from `lensml-plan.md` §13 on
> Day 28 of the build (README + landing page day). Until then, refer to
> the plan.

## Sections (planned)

1. What lensml is (one paragraph)
2. Why now (W&B / Neptune acquisitions; agentic coding maturity)
3. What's different (TrainCheck / AutoTrainer / Cursor / W&B comparison)
4. Architecture (ZMQ → FastAPI → SSE → Next.js)
5. What we don't do (yet) — DDP/FSDP, label leakage auto-fix, closed loops,
   production serving
6. Privacy mode
7. Benchmarks (lensml-bench)
8. License (Apache-2.0 core + `/ee/`)
