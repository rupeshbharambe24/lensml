"""lensml-bench — 48-fixture benchmark suite (plan §6).

12 pathology classes × 4 host environments = 48 broken training scripts.

Module map (plan §3.2):
    runner.py   subprocess orchestrator
    grader.py   per-bug deterministic checks
    report.py   markdown + json output
    fixtures/   the 48 broken-script directories (see fixtures/README.md)
"""
