"""Tiny lr-too-high fixture (bench plan §6.1).

Linear classifier on a clean separable problem with the learning rate
multiplied by 100×. Loss oscillates without improving — the classic
LR-too-high signal that ``LrTooHighDetector`` looks for.

Canonical fix: lower the learning rate by ~1 order of magnitude.
"""

from __future__ import annotations

import json

import torch
import torch.nn as nn

torch.manual_seed(0)

MAX_STEPS = 300
BATCH = 32
DIM = 8
N_CLASSES = 2

mu = torch.tensor([-2.0, 2.0])

model = nn.Linear(DIM, N_CLASSES)
optimizer = torch.optim.SGD(model.parameters(), lr=10.0)  # too high
criterion = nn.CrossEntropyLoss()

last_loss = float("nan")
for step in range(MAX_STEPS):
    y = torch.randint(0, N_CLASSES, (BATCH,))
    x = torch.randn(BATCH, DIM) * 0.5 + mu[y].unsqueeze(1)
    logits = model(x)
    loss = criterion(logits, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    last_loss = loss.item()

print(json.dumps({"final_loss": last_loss, "steps_used": MAX_STEPS}))
