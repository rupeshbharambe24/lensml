"""Tiny exploding-gradient fixture (bench plan §6.1).

A small MLP trained on synthetic-Gaussian data with a deliberately
oversized learning rate. With no gradient clipping, gradients explode
within a handful of steps and loss diverges to NaN.

Canonical fix: insert ``torch.nn.utils.clip_grad_norm_(model.parameters(),
max_norm=1.0)`` between ``loss.backward()`` and ``optimizer.step()``.
"""

from __future__ import annotations

import json

import torch
import torch.nn as nn

torch.manual_seed(0)

MAX_STEPS = 200
BATCH = 32
DIM = 8
N_CLASSES = 2

model = nn.Sequential(nn.Linear(DIM, 32), nn.ReLU(), nn.Linear(32, N_CLASSES))
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)  # too high
criterion = nn.CrossEntropyLoss()

last_loss = float("nan")
for step in range(MAX_STEPS):
    x = torch.randn(BATCH, DIM) * 5.0  # large inputs amplify the problem
    y = torch.randint(0, N_CLASSES, (BATCH,))
    logits = model(x)
    loss = criterion(logits, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    last_loss = loss.item()
    if step % 20 == 0:
        print(f"step={step} loss={last_loss:.4f}")

print(json.dumps({"final_loss": last_loss, "steps_used": MAX_STEPS}))
