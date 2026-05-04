"""Tiny dead-ReLU fixture (bench plan §6.1).

A 2-class MLP whose first layer is initialized with a large negative
bias so most ReLU outputs are pinned at zero from step 1. Gradients
can't flow through the dead units → the network can't learn the
class-conditional input distribution.

Canonical fix: replace ``nn.ReLU`` with ``nn.LeakyReLU(0.1)``. The
non-zero negative slope restores gradient flow through the dead units
and the optimizer recovers within the step budget.
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

# Class-conditional centers — without signal in the data, the optimum
# loss is just log(2) and *neither* arm could pass the threshold.
mu = torch.tensor([-0.5, 0.5])


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(DIM, 32)
        # Push the bias way negative so ReLU kills almost every unit.
        with torch.no_grad():
            self.fc1.bias.fill_(-2.0)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(32, N_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


model = Net()
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

print(json.dumps({"final_loss": last_loss, "steps_used": MAX_STEPS}))
