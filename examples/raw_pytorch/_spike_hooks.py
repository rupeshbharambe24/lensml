"""Day 0.4 hooks spike — see lensml-implementation-sequence.md.

Goal (from plan §2 / §8 Day 1):
    Confirm `register_forward_hook` + post-backward `param.grad` give us clean
    per-step telemetry with <5% overhead on a representative training step.

Plan-spec rationale (§2):
    - `register_forward_hook` and `register_full_backward_hook` are stable in
      PyTorch 2.x.
    - The deprecated `register_backward_hook` must be avoided.
    - Reading `param.grad` after `loss.backward()` is the canonical, low-
      overhead way to get per-parameter gradient stats — preferred over
      backward hooks.

Production-shape architecture (matches plan §3.1's 10–50 Hz tick rate):

    Two-layer split:
      - Hook (hot path, every step): just `out.detach()` — captures a
        reference to the latest forward output. No reductions, no Python
        conversions. Effectively free per-step.
      - Publish (cold path, every Nth step): walks captured tensors,
        computes summary stats (mean + dead_fraction on a strided sample),
        plus reads `param.grad` and computes global L2. This is the work
        that produces a ZMQ tick in real lensml.

    With PUBLISH_EVERY=5, on a 10ms/step CPU CNN, telemetry tick rate is
    20 Hz — within plan §3.1's 10–50 Hz envelope.

    On a real GPU training step (50–500ms), PUBLISH_EVERY=1 is fine:
    full reductions on GPU are bandwidth-bound and ~50× faster than CPU.

Hook captures only activation modules (plan §4 dead_relu detector + §5.2
inspect_activations). Other modules are inspected on-demand by agent tools.

Run:
    python examples/raw_pytorch/_spike_hooks.py

Exit codes:
    0  overhead < 5%   PASS
    1  overhead >= 5%  FAIL
"""

from __future__ import annotations

import statistics
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
N_SAMPLES = 2048
BATCH_SIZE = 64
STEPS = 100
TRIALS = 5
WARMUP = 10
SAMPLE_SIZE = 1024
PUBLISH_EVERY = 5  # plan §3.1 tick rate envelope (10–50 Hz); 5 = ~20 Hz at 10ms/step

ACTIVATION_TYPES: tuple[type, ...] = (
    nn.ReLU,
    nn.GELU,
    nn.Sigmoid,
    nn.Tanh,
    nn.LeakyReLU,
    nn.ELU,
    nn.SiLU,
)


def make_dataloader() -> DataLoader:
    g = torch.Generator().manual_seed(SEED)
    X = torch.randn(N_SAMPLES, 1, 28, 28, generator=g)
    y = torch.randint(0, 10, (N_SAMPLES,), generator=g)
    return DataLoader(TensorDataset(X, y), batch_size=BATCH_SIZE, shuffle=True)


class SmallCNN(nn.Module):
    """MNIST-class small CNN. Plan §6.2 'Tiny' benchmark environment shape."""

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


def _strided_sample(t: torch.Tensor, max_elements: int = SAMPLE_SIZE) -> torch.Tensor:
    flat = t.reshape(-1)
    n = flat.numel()
    if n <= max_elements:
        return flat
    return flat[:: n // max_elements]


class TelemetryBuffer:
    """Hook target: stores latest forward output per activation module.

    The hook is the hot path — must be as cheap as possible. It just does
    `out.detach()` (a tensor view, no copy) and stores a reference.
    """

    def __init__(self) -> None:
        self.activations: dict[str, torch.Tensor] = {}

    def make_hook(self, name: str):
        def hook(_module: nn.Module, _inp, out: torch.Tensor) -> None:
            if isinstance(out, torch.Tensor):
                self.activations[name] = out.detach()

        return hook


def install_capture_hooks(model: nn.Module, buf: TelemetryBuffer) -> list:
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, ACTIVATION_TYPES):
            handles.append(module.register_forward_hook(buf.make_hook(name)))
    return handles


def publish_telemetry(
    buf: TelemetryBuffer, model: nn.Module
) -> tuple[dict[str, dict[str, torch.Tensor]], torch.Tensor]:
    """Compute summary stats from the buffer + global grad-L2.

    This is the cold path — runs every PUBLISH_EVERY steps (plan §3.1 tick rate).
    """
    act_stats: dict[str, dict[str, torch.Tensor]] = {}
    for name, t in buf.activations.items():
        sample = _strided_sample(t)
        act_stats[name] = {
            "mean": sample.mean(),
            "dead_fraction": sample.eq(0).count_nonzero().float() / sample.numel(),
        }
    total_sq = torch.zeros((), device=DEVICE)
    for p in model.parameters():
        if p.grad is not None:
            total_sq = total_sq + p.grad.norm() ** 2
    return act_stats, total_sq.sqrt()


def make_model_and_opt() -> tuple[nn.Module, torch.optim.Optimizer]:
    torch.manual_seed(SEED)
    model = SmallCNN().to(DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    return model, opt


def stats_to_floats(stats: dict[str, dict[str, torch.Tensor]]) -> dict[str, dict[str, float]]:
    return {name: {k: float(v) for k, v in s.items()} for name, s in stats.items()}


def train_run(steps: int, with_hooks: bool, verbose: bool = False) -> tuple[float, dict, float]:
    model, opt = make_model_and_opt()
    loss_fn = nn.CrossEntropyLoss()
    dl = make_dataloader()

    buf = TelemetryBuffer()
    handles: list = []
    if with_hooks:
        handles = install_capture_hooks(model, buf)

    last_acts: dict = {}
    last_grad_l2 = 0.0
    it = iter(dl)
    t0 = time.perf_counter()
    for step in range(steps):
        try:
            xb, yb = next(it)
        except StopIteration:
            it = iter(dl)
            xb, yb = next(it)

        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()

        if with_hooks and step % PUBLISH_EVERY == 0:
            last_acts, grad_l2_t = publish_telemetry(buf, model)
            last_grad_l2 = float(grad_l2_t)
            if verbose and step % (PUBLISH_EVERY * 4) == 0:
                acts = stats_to_floats(last_acts)
                relu1 = acts.get("relu1", {})
                print(
                    f"  step {step:3d}  loss={loss.item():.4f}  "
                    f"grad-L2={last_grad_l2:.4f}  "
                    f"relu1.mean={relu1.get('mean', 0):.3f}  "
                    f"relu1.dead={relu1.get('dead_fraction', 0):.3f}"
                )

        opt.step()

    elapsed = time.perf_counter() - t0

    for h in handles:
        h.remove()

    return elapsed, last_acts, last_grad_l2


def verify_sampling_accuracy() -> None:
    torch.manual_seed(SEED)
    t = torch.relu(torch.randn(64, 16, 14, 14))
    full_mean = float(t.mean())
    full_dead = float((t == 0).float().mean())
    s = _strided_sample(t)
    sample_mean = float(s.mean())
    sample_dead = float(s.eq(0).count_nonzero().float() / s.numel())
    mean_err = abs(full_mean - sample_mean) / max(abs(full_mean), 1e-9) * 100
    dead_err = abs(full_dead - sample_dead) / max(full_dead, 1e-9) * 100
    print(f"sampling sanity (ReLU output, {t.numel()} elem -> {s.numel()} sample):")
    print(f"  full   mean={full_mean:+.4f}  dead={full_dead:.3f}")
    print(f"  sample mean={sample_mean:+.4f}  dead={sample_dead:.3f}")
    print(f"  rel error: mean={mean_err:.2f}%  dead={dead_err:.2f}%")
    print()


def main() -> int:
    print("=== lensml hooks spike ===")
    print(f"device: {DEVICE} | torch: {torch.__version__}")
    print("model:  SmallCNN (2 conv + 2 linear, MNIST shape)")
    print(f"steps={STEPS} batch={BATCH_SIZE} trials={TRIALS} warmup={WARMUP}")
    print("hooks:  capture-only (detach + store ref) on activation modules")
    print(f"publish: every {PUBLISH_EVERY} steps; strided sample={SAMPLE_SIZE}; global grad-L2")
    print()

    verify_sampling_accuracy()

    print(f"warmup ({WARMUP} steps, no hooks)... ", end="", flush=True)
    train_run(WARMUP, with_hooks=False)
    print("done.")
    print(f"warmup ({WARMUP} steps, with hooks)... ", end="", flush=True)
    train_run(WARMUP, with_hooks=True)
    print("done.")
    print()

    base_times: list[float] = []
    hook_times: list[float] = []
    last_acts: dict = {}
    last_grad: float = 0.0
    for trial in range(TRIALS):
        print(f"--- trial {trial + 1}/{TRIALS} ---")
        t_b, _, _ = train_run(STEPS, with_hooks=False)
        t_h, last_acts, last_grad = train_run(STEPS, with_hooks=True, verbose=(trial == 0))
        base_times.append(t_b)
        hook_times.append(t_h)
        print(f"  baseline:   {t_b * 1000:7.1f}ms  ({t_b * 1000 / STEPS:.3f}ms/step)")
        print(f"  with hooks: {t_h * 1000:7.1f}ms  ({t_h * 1000 / STEPS:.3f}ms/step)")
        print()

    # Use median of per-trial overhead percentages — more robust than
    # delta-of-medians (which can pair up unrelated noise across trials).
    per_trial_pct = [(h - b) / b * 100 for b, h in zip(base_times, hook_times, strict=True)]
    per_trial_delta_ms = [(h - b) * 1000 for b, h in zip(base_times, hook_times, strict=True)]
    overhead_pct = statistics.median(per_trial_pct)
    overhead_ms_step = statistics.median(per_trial_delta_ms) / STEPS
    base_med = statistics.median(base_times)
    hook_med = statistics.median(hook_times)
    print("=== per-trial overhead ===")
    for i, (b, h, pct, dms) in enumerate(
        zip(base_times, hook_times, per_trial_pct, per_trial_delta_ms, strict=True)
    ):
        print(
            f"  trial {i + 1}: baseline {b * 1000:7.1f}ms  hooks {h * 1000:7.1f}ms  delta {dms:+7.1f}ms  ({pct:+6.2f}%)"
        )
    print()

    print("=== final per-activation stats (last publish) ===")
    for name, s in stats_to_floats(last_acts).items():
        print(f"  {name:8s}  mean={s['mean']:+.4f}  dead_fraction={s['dead_fraction']:.3f}")
    print(f"  global grad-L2 (last publish): {last_grad:.4f}")
    print()

    print("=== overhead summary (median across trials) ===")
    print(f"  baseline median:    {base_med * 1000:7.1f}ms  ({base_med * 1000 / STEPS:.3f}ms/step)")
    print(f"  with hooks median:  {hook_med * 1000:7.1f}ms  ({hook_med * 1000 / STEPS:.3f}ms/step)")
    print(
        f"  delta:              {(hook_med - base_med) * 1000:+.1f}ms  ({overhead_ms_step:+.3f}ms/step)"
    )
    print(f"  overhead:           {overhead_pct:+.2f}%")
    print()

    if overhead_pct < 5.0:
        print(f"PASS: overhead {overhead_pct:.2f}% is below 5% target")
        return 0
    print(f"FAIL: overhead {overhead_pct:.2f}% exceeds 5% target")
    return 1


if __name__ == "__main__":
    sys.exit(main())
