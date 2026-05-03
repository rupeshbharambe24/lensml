"""Tiny training script for the watch.tape GIF.

Drops `lensml.watch(...)` onto a 2-layer MLP + synthetic data so the
dashboard at http://127.0.0.1:3000 receives live telemetry. The
script is intentionally short — just enough to demonstrate the
one-line API in the README.

Run with the lensml dashboard already up:
    lensml dashboard &
    python demo/scripts/demo_train.py
"""

from __future__ import annotations

import torch
import torch.nn as nn

import lensml

torch.manual_seed(0)

MAX_STEPS = 80
BATCH = 32
DIM = 8
N_CLASSES = 2

mu = torch.tensor([-1.0, 1.0])

model = nn.Sequential(nn.Linear(DIM, 32), nn.ReLU(), nn.Linear(32, N_CLASSES))
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
criterion = nn.CrossEntropyLoss()

# THE ONE LINE.
handle = lensml.watch(model, optimizer, run_id="demo-run")

try:
    for step in range(MAX_STEPS):
        y = torch.randint(0, N_CLASSES, (BATCH,))
        x = torch.randn(BATCH, DIM) * 0.5 + mu[y].unsqueeze(1)
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        handle.log(loss=loss.item())
        if step % 10 == 0:
            print(f"step={step:3d}  loss={loss.item():.4f}")
finally:
    handle.unwatch()
print("done.")
