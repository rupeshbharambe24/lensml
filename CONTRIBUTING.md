# Contributing to lensml

Thanks for your interest. lensml is in active development; contribution policy
will firm up after the v0.1 hackathon submission (June 7, 2026).

## Quick start (development setup)

```bash
git clone https://github.com/rupeshbharambe24/lensml
cd lensml
uv sync --all-extras
uv run pytest
uv run ruff check .
```

## Reporting bugs

Use the GitHub issue templates under `.github/ISSUE_TEMPLATE/`. A minimal
reproducer is the single most valuable thing you can include.

## Pull requests

- One change per PR.
- Include a test for new behavior.
- Run `uv run ruff check . && uv run ruff format .` before pushing.
- All commits must be signed off (DCO) — `git commit -s`.

## Code of conduct

This project follows the [Contributor Covenant 2.1](CODE_OF_CONDUCT.md).

## Where to start (good first issues)

- New pathology detector (see `lensml/detectors/` for examples)
- New libcst fix template (see `lensml/fixes/base.py` for the ABC)
- Benchmark fixtures (see `lensml/bench/fixtures/README.md`)
- Dashboard polish (see `dashboard/`)

## Questions?

Open a Discussion on GitHub or email software@biopanscientific.com.
