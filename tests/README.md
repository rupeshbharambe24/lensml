# Tests

Run with `uv run pytest -v`.

## Layout

```
tests/
  unit/          fast, no I/O, no network, no GPU
  integration/   FastAPI server, ZMQ transport, SQLite store
  fixtures/      shared pytest fixtures
```

## Required tests per build day (plan §8)

- Day 11: each libcst fix template has a unit test on a known input.
- Day 17–18: each detector has a unit test against a known telemetry stream.
- Day 12: end-to-end approval flow integration test.
- Day 13: checkpoint round-trip test (model + opt + scheduler + RNG + sampler).

## Markers

- `@pytest.mark.gpu` — skipped by default; run with `-m gpu` on CI with GPU runner.
- `@pytest.mark.slow` — skipped by default; run with `-m slow` for nightly.
- `@pytest.mark.network` — requires LLM provider keys; run locally only.
