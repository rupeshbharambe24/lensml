"""Per-fixture grader entry point — see lensml.bench.grader."""

from __future__ import annotations

import json
from pathlib import Path

from lensml.bench.fixtures import load_fixture
from lensml.bench.grader import grade_script

if __name__ == "__main__":
    fix = load_fixture(Path(__file__).parent)
    result = grade_script(Path(__file__).parent / "bug.py", fix, variant="bug")
    print(json.dumps(result.to_dict(), default=str, indent=2))
