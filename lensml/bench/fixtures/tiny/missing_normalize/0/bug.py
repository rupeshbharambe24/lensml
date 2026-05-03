"""Tiny missing-normalize fixture (bench plan §6.1).

A 2-class linear classifier on synthetic inputs whose values are in
``[0, 255]`` (faking raw uint8 image data). Without subtracting a mean
and dividing by a standard deviation, the inputs are far outside the
regime SGD is tuned for and the loss decays glacially.

Canonical fix: subtract mean and divide by std before the forward pass.
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

# Class-correlated unsigned data — each class has a different mean in the
# [0, 255] range. With normalization the linear model nails it; without,
# it crawls.
mu = torch.tensor([60.0, 195.0])

model = nn.Linear(DIM, N_CLASSES)
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
criterion = nn.CrossEntropyLoss()

last_loss = float("nan")
for step in range(MAX_STEPS):
    y = torch.randint(0, N_CLASSES, (BATCH,))
    x = torch.randn(BATCH, DIM) * 30.0 + mu[y].unsqueeze(1)
    x = x.clamp(0.0, 255.0)
    logits = model(x)
    loss = criterion(logits, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    last_loss = loss.item()

print(json.dumps({"final_loss": last_loss, "steps_used": MAX_STEPS}))
