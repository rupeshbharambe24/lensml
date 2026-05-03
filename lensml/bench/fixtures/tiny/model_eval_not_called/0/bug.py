"""Tiny model.eval()-not-called fixture (bench plan §6.1).

A small MLP with BatchNorm. The validation loop runs the model without
switching to eval mode — BatchNorm uses the *current batch* statistics
(garbage on a tiny val batch) instead of the running averages it
accumulated during training. Reported val accuracy is wildly off.

Canonical fix: wrap the validation loop with ``model.eval()`` /
``model.train()``.
"""

from __future__ import annotations

import json

import torch
import torch.nn as nn

torch.manual_seed(0)

MAX_STEPS = 200
BATCH = 32
DIM = 8
HID = 32
N_CLASSES = 2

mu = torch.tensor([-1.5, 1.5])


def make_batch(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    y = torch.randint(0, N_CLASSES, (n,))
    x = torch.randn(n, DIM) * 0.5 + mu[y].unsqueeze(1)
    return x, y


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(DIM, HID)
        self.bn = nn.BatchNorm1d(HID)
        self.fc2 = nn.Linear(HID, N_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.bn(self.fc1(x))))


def evaluate(model: nn.Module) -> float:
    # Bug: model is left in train() mode → BatchNorm uses current-batch stats.
    correct = 0
    total = 0
    for _ in range(5):
        x, y = make_batch(4)  # tiny val batches — BN stats are garbage
        with torch.no_grad():
            preds = model(x).argmax(dim=1)
        correct += int((preds == y).sum())
        total += len(y)
    return correct / total


model = Net()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()

last_loss = float("nan")
for step in range(MAX_STEPS):
    x, y = make_batch(BATCH)
    logits = model(x)
    loss = criterion(logits, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    last_loss = loss.item()

val_acc = evaluate(model)
print(json.dumps({"final_loss": last_loss, "final_metric": val_acc, "steps_used": MAX_STEPS}))
