"""Tiny lr-schedule-mismatch fixture (bench plan §6.1, stretch class).

A linear classifier trained with a fixed lr that's tuned for the
*early* phase of training. Without an LR schedule the optimizer keeps
overshooting near the minimum so loss plateaus well above what's
reachable; with cosine annealing the lr decays smoothly to a regime
where the optimizer can settle.

Canonical fix: wrap the optimizer in a ``CosineAnnealingLR`` scheduler
and step it after each ``optimizer.step()`` so the lr decays smoothly.
"""

from __future__ import annotations

import json

import torch
import torch.nn as nn

torch.manual_seed(0)

MAX_STEPS = 400
BATCH = 32
DIM = 8
N_CLASSES = 2

mu = torch.tensor([-0.3, 0.3])

model = nn.Linear(DIM, N_CLASSES)
optimizer = torch.optim.SGD(model.parameters(), lr=2.0)
criterion = nn.CrossEntropyLoss()

last_loss = float("nan")
for step in range(MAX_STEPS):
    y = torch.randint(0, N_CLASSES, (BATCH,))
    # Overlapping clusters → reachable optimum is meaningful (~0.1).
    # With a fixed lr=2.0 the optimizer keeps overshooting near the
    # minimum and plateaus above the threshold.
    x = torch.randn(BATCH, DIM) * 0.4 + mu[y].unsqueeze(1)
    logits = model(x)
    loss = criterion(logits, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    last_loss = loss.item()

print(json.dumps({"final_loss": last_loss, "steps_used": MAX_STEPS}))
