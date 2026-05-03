"""Tiny bad-init fixture (bench plan §6.1).

A 2-class linear classifier whose weights are initialized with a much
larger scale than the default (``std=10`` instead of the default
~``1/sqrt(fan_in)``). Step-0 cross-entropy is wildly above the
random-baseline ``log(2) ≈ 0.69`` and convergence is extremely slow.

Canonical fix: re-initialize weights with Kaiming-normal (the standard
init for ReLU/Linear layers) before training.
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

mu = torch.tensor([-1.5, 1.5])

model = nn.Linear(DIM, N_CLASSES)
# Bug: huge weights = confident-wrong predictions = enormous CE at init.
with torch.no_grad():
    model.weight.normal_(mean=0.0, std=10.0)

optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
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

print(json.dumps({"final_loss": last_loss, "steps_used": MAX_STEPS, "num_classes": N_CLASSES}))
