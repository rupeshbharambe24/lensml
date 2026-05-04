"""Tiny model.eval()-not-called fixture (bench plan §6.1).

A small MLP with BatchNorm + Dropout. The validation loop runs the
model without switching to eval mode, so:

    1. BatchNorm uses the *current val batch* statistics instead of
       the running averages it accumulated during training.
    2. Dropout fires randomly, dropping half the activations during
       what's supposed to be deterministic inference.

Both effects together drive val accuracy down to roughly chance.

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

mu = torch.tensor([-0.4, 0.4])


def make_batch(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    y = torch.randint(0, N_CLASSES, (n,))
    x = torch.randn(n, DIM) * 0.4 + mu[y].unsqueeze(1)
    return x, y


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(DIM, HID)
        self.bn = nn.BatchNorm1d(HID)
        self.dropout = nn.Dropout(0.8)
        self.fc2 = nn.Linear(HID, N_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(torch.relu(self.bn(self.fc1(x)))))


def evaluate(model: nn.Module) -> float:
    # Bug: model is left in train() mode → BatchNorm uses current-batch
    # stats AND Dropout is still active during what should be inference.
    correct = 0
    total = 0
    for _ in range(20):
        x, y = make_batch(8)
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
