# lensml-bench

A reproducibility benchmark for the v1 detector and fix-template
catalogue (plan §6).

## Layout

```
lensml/bench/
  fixtures.py           Fixture loader + meta.json validator
  grader.py             run_script + grade_script + apply_diff
  runner.py             BenchRunner: run_one + run_all + JSON cache
  baseline.py           BaselineRunner (LLM-only baseline arm)
  report.py             Markdown + JSON report renderer
  fixtures/
    tiny/<class>/<idx>/
      bug.py            broken training script
      fix.diff          canonical unified diff
      grade.py          grader entry-point shim
      meta.json         {pathology_class, difficulty, host_env, max_steps,
                         loss_threshold, metric_threshold?, metric_name?}
```

## Three-check protocol (plan §6.1)

1. **Runs** — does the (patched) script exit 0?
2. **Loss converged** — does `final_loss <= meta.loss_threshold`?
3. **Metric passed** — when `meta.metric_threshold` is set, does
   `final_metric >= threshold`?

Scripts emit a final-line JSON status:

```python
print(json.dumps({"final_loss": 0.12, "final_metric": 0.91, "steps_used": 200}))
```

The grader parses the last `{...}` line, so any mid-run prints are
fine.

## What `lensml bench run` does

For each fixture:

1. Subprocess `bug.py`. Grade. Expected: ≥ one check fails.
2. Apply `fix.diff` to `bug.py` → `patched.py` in a tmp dir.
3. Subprocess `patched.py`. Grade. Expected: all checks pass.
4. Record both as a `FixtureRun` (with `bug_correctly_fails`,
   `fix_passes`, `overall_pass` derived properties).

JSON cache (`--cache PATH`) lets you resume after Colab/Kaggle
disconnects (plan §11 risk #6).

## The LLM-only baseline arm

`lensml bench baseline --provider <name>` runs every fixture
through a single-shot prompt: "diagnose this and produce a unified
diff." `parse_diagnosis_class` maps the LLM's prose to one of the
12 canonical detector slugs; `parse_diff_from_response` extracts the
fenced ```diff …``` block. The diff is applied + graded the same
way `lensml bench run` grades the canonical patch.

This is the credibility lynchpin (plan §6.3): if lensml-full doesn't
beat single-shot LLM diagnosis on the same fixtures, the integration
story doesn't hold up.

## Final report

`lensml bench report --bench JSON --baseline JSON … --out DIR`
stitches arm-specific JSONs into one `report.md` with:

- Headline (overall pass / fix-pass / bug-fails counts).
- lensml-full vs LLM-only comparison table.
- Confusion matrix per baseline arm (diagonal bolded).
- Latency histogram across all arms (shared bucket edges).
- Cost breakdown — every arm at $0.00.
- Per-pathology aggregate.
- Per-fixture pass/fail.
- "What doesn't work yet" — fixtures whose `patched` failed.

## Adding a fixture

1. `mkdir -p lensml/bench/fixtures/tiny/<class>/<idx>/`.
2. Write `bug.py` (must reliably fail at least one of the three
   checks).
3. Write the fixed version manually, then generate the diff via
   `difflib.unified_diff` (see the `regenerate fix.diff` snippet in
   any of the existing fixtures' commit history).
4. Write a `grade.py` shim (3 lines — see existing fixtures).
5. Write `meta.json` with thresholds chosen so the bug fails and
   the fix passes.
6. Run `lensml bench run --pathology <class>` to verify.
