"""Tiny exploding-gradient fixture (bench plan §6.1).

A small MLP trained on class-correlated synthetic data with a
deliberately oversized learning rate and amplified inputs. With no
gradient clipping, the loss either oscillates wildly above the
achievable optimum or diverges to NaN — never settles below the
``loss_threshold``.

Canonical fix: lower the LR to 0.05 and insert
``torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)``
between ``loss.backward()`` and ``optimizer.step()``. Loss converges
cleanly below the threshold.
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

# Class-correlated centers so the optimum is well below log(2).
mu = torch.tensor([-1.0, 1.0])

model = nn.Sequential(nn.Linear(DIM, 32), nn.ReLU(), nn.Linear(32, N_CLASSES))
optimizer = torch.optim.SGD(model.parameters(), lr=5.0)  # too high → diverges
criterion = nn.CrossEntropyLoss()

last_loss = float("nan")
for step in range(MAX_STEPS):
    y = torch.randint(0, N_CLASSES, (BATCH,))
    x = (torch.randn(BATCH, DIM) + mu[y].unsqueeze(1)) * 5.0  # amplify
    logits = model(x)
    loss = criterion(logits, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    last_loss = loss.item()
    if step % 20 == 0:
        print(f"step={step} loss={last_loss:.4f}")

print(json.dumps({"final_loss": last_loss, "steps_used": MAX_STEPS}))
