"""Per-fixture grader entry point. Thin shim — real logic lives in
``lensml.bench.grader``. Provided so the fixture directory carries the
file name advertised in the plan §6.1 file layout."""

from __future__ import annotations

from pathlib import Path

from lensml.bench.fixtures import load_fixture
from lensml.bench.grader import grade_script

if __name__ == "__main__":
    fix = load_fixture(Path(__file__).parent)
    result = grade_script(Path(__file__).parent / "bug.py", fix, variant="bug")
    import json

    print(json.dumps(result.to_dict(), default=str, indent=2))
