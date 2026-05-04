"""Tiny lr-too-high fixture (bench plan §6.1).

Linear classifier on a clean separable problem with the learning rate
~3 orders of magnitude too large. Loss oscillates without converging
— the classic LR-too-high signal that ``LrTooHighDetector`` looks for.

The grader looks at the *peak loss in the trailing window* (not just
the last step or mean). Oscillation always has a big spike in any
window even when individual steps drift near the optimum.

Canonical fix: lower the learning rate by ~2 orders of magnitude.
"""

from __future__ import annotations

import json
from collections import deque

import torch
import torch.nn as nn

torch.manual_seed(0)

MAX_STEPS = 200
BATCH = 32
DIM = 8
N_CLASSES = 2

mu = torch.tensor([-0.4, 0.4])

model = nn.Linear(DIM, N_CLASSES)
optimizer = torch.optim.SGD(model.parameters(), lr=50.0)  # too high → oscillates
criterion = nn.CrossEntropyLoss()

window: deque[float] = deque(maxlen=20)
for step in range(MAX_STEPS):
    y = torch.randint(0, N_CLASSES, (BATCH,))
    x = torch.randn(BATCH, DIM) * 0.4 + mu[y].unsqueeze(1)
    logits = model(x)
    loss = criterion(logits, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    window.append(loss.item())

# Peak of the last 20 steps — an oscillating loss always has a big
# spike in any trailing window even when individual steps land near
# the optimum.
final_loss = max(window) if window else float("nan")
print(json.dumps({"final_loss": final_loss, "steps_used": MAX_STEPS}))
