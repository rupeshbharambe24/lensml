# lensml docs

Built with [mkdocs-material](https://squidfunk.github.io/mkdocs-material/).

## Structure

```
docs/
  index.md             landing page
  DESIGN.md            public design doc (plan §13)
  install.md           installation
  quickstart.md        30-second tour
  integrations/        per-framework guides
    pytorch.md
    lightning.md
    transformers.md
  pathologies/         the 8 v1 pathology classes (plan §4)
  bench/               lensml-bench documentation
  privacy.md           local privacy mode (plan §5.6)
  faq.md
```

## Build locally

```bash
uv run --extra docs mkdocs serve
```

## Deploy

For v1 (free-tier only): docs site auto-builds to GitHub Pages at
`https://rupeshbharambe24.github.io/lensml/` on push to `main`.
The Next.js landing page (separate from this docs site) is in `dashboard/`
and deploys to Vercel's free tier (default URL `https://lensml.vercel.app`).

Custom domain (`lensml.dev`) is deferred until post-Round 1; the URL is a
single DNS swap at any time.
