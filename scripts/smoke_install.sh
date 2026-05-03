#!/usr/bin/env bash
# Fresh-environment install smoke test (plan §8 step 5.1).
#
# Run inside the Dockerfile or any clean Python 3.11 venv. Validates
# that the v1 install path works without depending on dev machine
# state — the question the Unstop reviewer is asking when they spin
# up the container.
#
# Exits 0 on success, non-zero with a banner on first failure.

set -euo pipefail

banner() {
    printf "\n=== %s ===\n" "$1"
}

banner "Python + pip versions"
python --version
pip --version

banner "Import lensml"
python -c "import lensml; print('OK lensml:', lensml.__name__)"

banner "lensml CLI surface"
lensml --help

banner "Probe core submodules"
python <<'PY'
modules = [
    "lensml.core.signals",
    "lensml.core.transport",
    "lensml.detectors.base",
    "lensml.detectors.exploding_grad",
    "lensml.detectors.dead_relu",
    "lensml.detectors.lr_too_high",
    "lensml.detectors.missing_normalize",
    "lensml.detectors.bad_init",
    "lensml.detectors.vanishing_grad",
    "lensml.detectors.model_eval_not_called",
    "lensml.detectors.dataloader_bottleneck",
    "lensml.fixes",
    "lensml.replay",
    "lensml.bench.fixtures",
    "lensml.bench.grader",
    "lensml.bench.runner",
    "lensml.bench.baseline",
    "lensml.server.app",
    "lensml.server.privacy",
]
for m in modules:
    __import__(m)
    print("OK", m)
PY

banner "Probe FIX_REGISTRY"
python -c "
from lensml.fixes import FIX_REGISTRY
expected = {
    'AddGradientClipping', 'LowerLearningRate', 'ReplaceActivation',
    'AddNormalize', 'InsertEvalMode', 'KaimingInit',
    'DataLoaderWorkers', 'AddLRScheduler',
}
got = set(FIX_REGISTRY.keys())
missing = expected - got
assert not missing, f'missing fix templates: {missing}'
print('OK FIX_REGISTRY:', len(got), 'templates')
"

banner "Probe packaged bench fixtures"
python -c "
from lensml.bench.fixtures import discover_fixtures
fxs = discover_fixtures()
assert len(fxs) >= 8, f'expected at least 8 packaged fixtures, got {len(fxs)}'
slugs = {f.slug for f in fxs}
print('OK fixtures:', len(fxs))
for s in sorted(slugs):
    print('   -', s)
"

banner "Probe replay scenarios"
python -c "
from lensml.replay.scenarios import list_scenarios, generate
names = list_scenarios()
assert set(names) == {'exploding_grad', 'dead_relu', 'missing_normalize'}, names
for n in names:
    events = generate(n)
    assert len(events) > 5, f'{n} generated {len(events)} events'
    print('OK scenario:', n, '->', len(events), 'events')
"

banner "All smoke checks passed"
