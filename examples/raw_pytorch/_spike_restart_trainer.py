"""Day 0.6 subprocess-restart spike — trainer half.

The orchestrator (_spike_restart_orchestrator.py) runs this trainer in three
configurations and compares loss trajectories.

Modes (controlled by CLI args):
    1. Fresh-start uninterrupted: --output A.jsonl --total-steps 12
    2. Fresh-start with kill mid-run: --output B.jsonl --total-steps 12
       --kill-at 4 --ckpt-out C.pt   (exits with code 99 after checkpointing)
    3. Resume from checkpoint:    --output B.jsonl --total-steps 12
       --resume C.pt   (appends remaining losses)

Plan §3.3 checkpoint contents tested HERE (the v1 critical path):
    - model.state_dict()        ✓
    - optimizer.state_dict()    ✓
    - torch RNG state (CPU+CUDA) ✓
    - numpy + python random states ✓
    - step number               ✓

Plan §3.3 explicitly notes sampler.pt as "best effort". DataLoader sampler
internals (especially `_BaseDataLoaderIter`'s extra `torch.empty().random_()`
call for worker base-seed, plus the BatchSampler/RandomSampler interaction
with `num_samples == n` triggering a phantom 2nd randperm) make byte-exact
sampler state matching nontrivial. For this spike we sidestep by using
`shuffle=False` with a pre-shuffled deterministic dataset — the data ordering
is encoded in the dataset, not the sampler. Production lensml will need a
custom Sampler that exposes a clean state_dict() — v0.2 concern.

Determinism setup (must happen before any heavy torch import):
    - CUBLAS_WORKSPACE_CONFIG=:4096:8 (CUDA determinism)
    - torch.use_deterministic_algorithms(True, warn_only=True)
    - cudnn.deterministic = True; cudnn.benchmark = False
    - torch.set_num_threads(1) (CPU determinism)
    - All RNG seeds locked
"""

from __future__ import annotations

import os

# Must be set before torch is imported
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
DATA_SEED = 7
SHUFFLE_SEED = 123  # used to permute the dataset once at startup

DATASET_SIZE = 64
BATCH_SIZE = 16
# 4 batches per epoch.


def set_seed(seed: int) -> None:
    """Lock all RNG sources for determinism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(1)


class SmallCNN(nn.Module):
    """Same architecture as the hooks spike — keeps the spike pair coherent."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        return self.fc2(x)


def make_dataset() -> TensorDataset:
    """Fixed-seed synthetic data, pre-permuted once with SHUFFLE_SEED.

    Pre-shuffling the data lets us use shuffle=False in the DataLoader, which
    sidesteps sampler-state checkpoint complexity (plan §3.3 'best effort').
    """
    g_data = torch.Generator().manual_seed(DATA_SEED)
    X = torch.randn(DATASET_SIZE, 1, 28, 28, generator=g_data)
    y = torch.randint(0, 10, (DATASET_SIZE,), generator=g_data)

    # One-shot deterministic shuffle — same in every run because seed is fixed.
    g_shuf = torch.Generator().manual_seed(SHUFFLE_SEED)
    perm = torch.randperm(DATASET_SIZE, generator=g_shuf)
    X = X[perm]
    y = y[perm]
    return TensorDataset(X, y)


def save_checkpoint(
    path: Path,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    next_batch_index: int,
) -> None:
    """Per plan §3.3 (v1 critical-path subset): model + optimizer + RNG + step.

    `next_batch_index` records the position in the (deterministic, pre-shuffled)
    dataset to resume from. Sidesteps sampler.pt complexity per plan §3.3.
    """
    ckpt = {
        "step": step,
        "next_batch_index": next_batch_index,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
    }
    if torch.cuda.is_available():
        ckpt["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    torch.save(ckpt, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, int]:
    """Returns (step, next_batch_index)."""
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    torch.set_rng_state(ckpt["torch_rng_state"])
    if "cuda_rng_state" in ckpt and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])
    random.setstate(ckpt["python_rng_state"])
    np.random.set_state(ckpt["numpy_rng_state"])
    return ckpt["step"], ckpt["next_batch_index"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--total-steps", type=int, default=12)
    p.add_argument("--kill-at", type=int, default=None)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--ckpt-out", type=Path, default=Path("ckpt.pt"))
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = p.parse_args()

    set_seed(SEED)
    device = torch.device(args.device)

    model = SmallCNN().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    if args.resume:
        start_step, next_batch_index = load_checkpoint(args.resume, model, optimizer)
        sys.stderr.write(
            f"[trainer] resumed from step {start_step}, next_batch={next_batch_index}\n"
        )
    else:
        start_step = 0
        next_batch_index = 0
        sys.stderr.write(f"[trainer] fresh start on {device}\n")

    loss_fn = nn.CrossEntropyLoss()
    ds = make_dataset()
    # shuffle=False — data is pre-shuffled deterministically in make_dataset.
    # We track the batch index manually so resume can fast-forward to the right spot.
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    mode = "a" if args.resume else "w"
    out = args.output.open(mode)

    step = start_step
    batches_per_epoch = len(dl)  # = 4 for 64/16
    it = iter(dl)
    # Fast-forward to next_batch_index within the current epoch
    epoch_pos = next_batch_index % batches_per_epoch
    for _ in range(epoch_pos):
        next(it)
    while step < args.total_steps:
        try:
            xb, yb = next(it)
        except StopIteration:
            it = iter(dl)
            xb, yb = next(it)

        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()

        # Diagnostic fingerprints to localize divergence
        with torch.no_grad():
            xb_sum = float(xb.sum())
            yb_sum = int(yb.sum())
            wsum = float(sum(p.sum() for p in model.parameters()))
        record = {
            "step": step,
            "loss": loss.item(),
            "xb_sum": xb_sum,
            "yb_sum": yb_sum,
            "wsum": wsum,
        }
        out.write(json.dumps(record) + "\n")
        out.flush()
        step += 1

        if args.kill_at is not None and step == args.kill_at:
            save_checkpoint(args.ckpt_out, step, model, optimizer, next_batch_index=step)
            out.close()
            sys.stderr.write(f"[trainer] checkpointed at step {step}, exiting with code 99\n")
            return 99

    out.close()
    sys.stderr.write(f"[trainer] completed {step} steps\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
