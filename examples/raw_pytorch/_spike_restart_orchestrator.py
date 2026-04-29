"""Day 0.6 subprocess-restart spike — orchestrator.

Runs the trainer (`_spike_restart_trainer.py`) in three configurations:
    A.  Uninterrupted control: full 12-step run.
    B-1. Kill mid-run: 12-step plan, kill+checkpoint at step 4.
    B-2. Resume: load checkpoint, run remaining 8 steps.

Then compares the loss trajectories of A vs B (B-1 + B-2 concatenated).
Goal: byte-identical losses, proving plan §3.3 checkpoint contents
preserve full state continuity (model + optimizer + RNG + sampler).

Plan §11 risk #1: subprocess restart with full state continuity is one of
the highest-risk items in the build calendar (Day 13). This spike de-risks
it before the build window opens.

Run:
    python examples/raw_pytorch/_spike_restart_orchestrator.py [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
TRAINER = HERE / "_spike_restart_trainer.py"
OUT_DIR = HERE / "_restart_outputs"
OUT_DIR.mkdir(exist_ok=True)

LOSSES_A = OUT_DIR / "losses_a.jsonl"
LOSSES_B = OUT_DIR / "losses_b.jsonl"
CKPT_B = OUT_DIR / "ckpt_b.pt"

TOTAL_STEPS = 12
KILL_AT = 4  # end of epoch 1 (4 batches/epoch with batch=16, ds=64)


def run_trainer(args: list[str], expected_exit: int) -> None:
    cmd = [sys.executable, str(TRAINER), *args]
    print(f"  $ {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.stderr:
        for line in r.stderr.rstrip().splitlines():
            print(f"    {line}")
    if r.returncode != expected_exit:
        print(f"  unexpected exit code {r.returncode} (expected {expected_exit})")
        if r.stdout:
            print("  stdout:", r.stdout[:500])
        sys.exit(1)


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def cleanup() -> None:
    for p in (LOSSES_A, LOSSES_B, CKPT_B):
        if p.exists():
            p.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    print("=== lensml subprocess restart spike ===")
    print(f"trainer:      {TRAINER}")
    print(f"python:       {sys.executable}")
    print(f"device:       {args.device}")
    print(f"total steps:  {TOTAL_STEPS}, kill at: {KILL_AT}")
    print()

    cleanup()

    print("--- Run A: uninterrupted control ---")
    run_trainer(
        [
            "--output",
            str(LOSSES_A),
            "--total-steps",
            str(TOTAL_STEPS),
            "--device",
            args.device,
        ],
        expected_exit=0,
    )
    losses_a = read_jsonl(LOSSES_A)
    print(f"  produced {len(losses_a)} loss entries")
    print()

    print(f"--- Run B-1: train + kill at step {KILL_AT} ---")
    run_trainer(
        [
            "--output",
            str(LOSSES_B),
            "--total-steps",
            str(TOTAL_STEPS),
            "--kill-at",
            str(KILL_AT),
            "--ckpt-out",
            str(CKPT_B),
            "--device",
            args.device,
        ],
        expected_exit=99,
    )
    losses_b1 = read_jsonl(LOSSES_B)
    print(f"  produced {len(losses_b1)} loss entries before kill")
    print(f"  checkpoint: {CKPT_B} ({CKPT_B.stat().st_size:,} bytes)")
    print()

    print("--- Run B-2: resume from checkpoint ---")
    run_trainer(
        [
            "--output",
            str(LOSSES_B),
            "--total-steps",
            str(TOTAL_STEPS),
            "--resume",
            str(CKPT_B),
            "--device",
            args.device,
        ],
        expected_exit=0,
    )
    losses_b = read_jsonl(LOSSES_B)
    print(f"  total {len(losses_b)} loss entries (B-1 + B-2 concatenated)")
    print()

    print("=== comparison: A vs (B-1 + B-2) ===")
    if len(losses_a) != len(losses_b):
        print(f"FAIL: length mismatch ({len(losses_a)} vs {len(losses_b)})")
        return 1

    print(
        f"{'step':>4}  {'loss_A':>13}  {'loss_B':>13}  {'xb_A':>12}  {'xb_B':>12}  {'wsum_A':>13}  {'wsum_B':>13}  loss"
    )
    n_exact = 0
    max_abs_diff = 0.0
    for a, b in zip(losses_a, losses_b, strict=True):
        if a["step"] != b["step"]:
            print(f"FAIL: step mismatch at index {a['step']} vs {b['step']}")
            return 1
        diff = abs(a["loss"] - b["loss"])
        max_abs_diff = max(max_abs_diff, diff)
        loss_eq = "EXACT" if a["loss"] == b["loss"] else f"d={diff:.2e}"
        if a["loss"] == b["loss"]:
            n_exact += 1
        xb_eq = "=" if a["xb_sum"] == b["xb_sum"] else "X"
        w_eq = "=" if a["wsum"] == b["wsum"] else "X"
        print(
            f"{a['step']:>4}  {a['loss']:>13.8f}  {b['loss']:>13.8f}  "
            f"{a['xb_sum']:>12.4f}  {b['xb_sum']:>12.4f}{xb_eq} "
            f"{a['wsum']:>13.6f}  {b['wsum']:>13.6f}{w_eq} {loss_eq}"
        )
    print()

    print(f"exact matches: {n_exact}/{len(losses_a)}")
    print(f"max abs diff:  {max_abs_diff:.3e}")
    print()

    if n_exact == len(losses_a):
        print("PASS: byte-identical loss trajectories across kill+restart")
        print("      -> plan §3.3 checkpoint contents are sufficient for full state continuity")
        return 0
    if max_abs_diff < 1e-6:
        print(f"NEAR-PASS: trajectories match within FP tolerance ({max_abs_diff:.2e})")
        print("           -> state continuity preserved; minor non-determinism in ops")
        return 0
    print("FAIL: trajectories diverge beyond FP tolerance")
    print("      -> some state isn't being captured/restored correctly")
    return 1


if __name__ == "__main__":
    sys.exit(main())
