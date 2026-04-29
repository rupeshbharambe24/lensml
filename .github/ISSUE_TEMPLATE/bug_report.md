---
name: Bug report
about: Report a bug or unexpected behavior in lensml
title: "[BUG] "
labels: bug
assignees: ""
---

**Describe the bug**
A clear and concise description of what the bug is.

**Reproducer**
Minimal training script (or Lightning / HF Trainer config) that triggers the issue.

```python
import lensml
# ...
```

**Expected behavior**
What you expected to happen.

**Actual behavior**
What actually happened. Include any agent diagnosis output if relevant.

**Environment**
- lensml version: `pip show lensml | grep Version`
- Python: `python --version`
- PyTorch: `python -c "import torch; print(torch.__version__)"`
- OS:
- GPU / VRAM:
- LLM provider in use:

**Additional context**
Logs from `~/.lensml/runs/{run_id}/events.db` if reproducible.
