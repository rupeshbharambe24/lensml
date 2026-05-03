# Links — single source of truth for the submission form

Update each `TBD` line as you generate the asset, then copy-paste
into the Unstop fields.

## Code

- **GitHub repo (public on June 7 PM):** https://github.com/rupeshbharambe24/lensml
- **License (core):** Apache-2.0 (see `LICENSE`)
- **License (enterprise scaffolding):** see `LICENSE-EE`
- **Reproducible install:** `docker build -t lensml-smoke . && docker run --rm lensml-smoke`

## Sites

- **Landing page (Vercel free tier):** https://lensml.vercel.app
- **Docs (GitHub Pages free tier):** https://rupeshbharambe24.github.io/lensml/

## Demo

- **90-second video (YouTube unlisted):** _TBD — see `demo/RECORDING.md`_
- **Hero GIFs (regenerable):** `demo/install.gif`, `demo/watch.gif`, `demo/intervene.gif`
- **Live demo path (no LLM, no GPU):** `lensml replay --scenario exploding_grad --speed 2`

## Benchmark

- **Bench source:** `lensml/bench/`
- **Final report (after external Colab/Kaggle run):** _TBD — `bench/results/2026-06-XX/report.md`_
- **Reproduce:** `lensml bench run --out bench/results/<date>/`
- **LLM-only baseline arm:** `lensml bench baseline --provider groq --out bench/results/<date>/`
- **Stitch:** `lensml bench report --bench …/report.json --baseline …/baseline-groq.json --out …`

## Build provenance

- **Day-by-day build log:** [lensml-implementation-sequence.md](../lensml-implementation-sequence.md)
- **Public design doc:** [docs/DESIGN.md](../docs/DESIGN.md)
- **30-day plan that drove it:** `lensml-plan.md` (kept private until R2 results)

## Author

- **Solo developer:** Rupesh Bharambe (@rupeshbharambe24)
- **Email:** software@biopanscientific.com
- **Tech stack:** Python 3.11, FastAPI, Next.js 15, libcst, LiteLLM, PyTorch
