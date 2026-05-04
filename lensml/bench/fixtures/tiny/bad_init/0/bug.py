"""Tiny bad-init fixture (bench plan §6.1).

A 2-class linear classifier whose weights are initialized with a much
larger scale than the default (``std=10`` instead of the default
~``1/sqrt(fan_in)``). With a tight step budget the optimizer cannot
recover from those huge weights, so the bug.py never reaches the loss
threshold. The patched version (Kaiming-normal init) lands close to
the optimum within a handful of steps.

Canonical fix: re-initialize weights with Kaiming-normal before training.
"""

from __future__ import annotations

import json

import torch
import torch.nn as nn

torch.manual_seed(0)

MAX_STEPS = 30
BATCH = 32
DIM = 8
N_CLASSES = 2

# Tight clusters with low-amplitude separation — the optimizer needs
# every step to recover from a bad init. Patched (kaiming_normal_)
# starts close to zero so all 30 steps are productive descent.
mu = torch.tensor([-0.4, 0.4])

model = nn.Linear(DIM, N_CLASSES)
# Bug: huge weights = confident-wrong predictions = enormous CE at init.
with torch.no_grad():
    model.weight.normal_(mean=0.0, std=10.0)

optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
criterion = nn.CrossEntropyLoss()

last_loss = float("nan")
for step in range(MAX_STEPS):
    y = torch.randint(0, N_CLASSES, (BATCH,))
    x = torch.randn(BATCH, DIM) * 0.4 + mu[y].unsqueeze(1)
    logits = model(x)
    loss = criterion(logits, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    last_loss = loss.item()

print(json.dumps({"final_loss": last_loss, "steps_used": MAX_STEPS, "num_classes": N_CLASSES}))
