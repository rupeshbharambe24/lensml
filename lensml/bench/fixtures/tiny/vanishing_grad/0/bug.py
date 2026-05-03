"""Tiny vanishing-gradient fixture (bench plan §6.1).

A 6-layer MLP with sigmoid activations on a small classification task.
Sigmoids saturate, gradients shrink exponentially with depth, and the
deepest layer's gradient is effectively zero — training stalls.

Canonical fix: replace sigmoids with GELU (or any non-saturating
activation). Gradient magnitudes propagate cleanly through depth.
"""

from __future__ import annotations

import json

import torch
import torch.nn as nn

torch.manual_seed(0)

MAX_STEPS = 400
BATCH = 32
DIM = 8
HID = 16
N_CLASSES = 2

mu = torch.tensor([-1.0, 1.0])


class DeepNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(DIM, HID),
            nn.Sigmoid(),
            nn.Linear(HID, HID),
            nn.Sigmoid(),
            nn.Linear(HID, HID),
            nn.Sigmoid(),
            nn.Linear(HID, HID),
            nn.Sigmoid(),
            nn.Linear(HID, HID),
            nn.Sigmoid(),
            nn.Linear(HID, N_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


model = DeepNet()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
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
