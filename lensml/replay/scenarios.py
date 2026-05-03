"""Canned demo scenarios for ``lensml replay`` (plan §3.6, §8 D20).

Three deterministic generators that synthesize a realistic-looking
training trace ending in a specific pathology. Each generator yields
the same kind of event dicts a live trainer would publish (run_started
→ metrics → anomaly → optionally proposed_fix → run_ended), so the
scenarios drive the entire downstream pipeline (broker → SSE →
detectors? → agent? → dashboard) just like a real run.

Detectors are ALSO server-side stream consumers (step 3.3/3.4), so we
emit our own ``kind: "anomaly"`` events directly rather than relying
on the detector loop being wired up — this guarantees the dashboard
shows the intervention markers + agent reasoning pane even if the
LLM tier is offline.

Scenarios:
    exploding_grad        — grad-L2 spike → grad-clip fix proposal
    dead_relu             — dead ReLU climbs → replace_activation fix
    missing_normalize     — first-batch input_stats triggers fix

Each ``generate_*`` returns ``list[dict]`` (NOT a generator) so the
scenarios are easy to count, snapshot-test, and tee to JSONL.
"""

from __future__ import annotations

import math
import random
import time
from typing import Any

DEFAULT_PRE_STEPS = 60
DEFAULT_POST_STEPS = 30


def _step_event(
    *,
    run_id: str,
    step: int,
    ts: float,
    loss: float,
    grad_l2: float,
    lr: float = 0.01,
    activations: dict[str, dict[str, float]] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "kind": "metric",
        "run_id": run_id,
        "step": step,
        "ts": ts,
        "loss": loss,
        "grad_l2": grad_l2,
        "lr": lr,
    }
    if activations is not None:
        payload["activations"] = activations
    if extra:
        payload.update(extra)
    return payload


def _run_started(run_id: str, ts: float) -> dict[str, Any]:
    return {
        "kind": "run_started",
        "run_id": run_id,
        "ts": ts,
        "step": 0,
        "resumed": False,
    }


def _run_ended(run_id: str, ts: float, step: int) -> dict[str, Any]:
    return {
        "kind": "run_ended",
        "run_id": run_id,
        "ts": ts,
        "step": step,
    }


def _anomaly(
    *,
    run_id: str,
    step: int,
    ts: float,
    detector: str,
    signal: str,
    severity: str,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    return {
        "kind": "anomaly",
        "run_id": run_id,
        "step": step,
        "ts": ts,
        "detector": detector,
        "signal": signal,
        "severity": severity,
        "evidence": evidence,
    }


# ---- exploding_grad ----------------------------------------------------------


def generate_exploding_grad(
    *,
    run_id: str = "demo-exploding-grad",
    pre_steps: int = DEFAULT_PRE_STEPS,
    post_steps: int = DEFAULT_POST_STEPS,
    base_ts: float = 0.0,
    step_dt: float = 0.05,
    seed: int = 1,
) -> list[dict[str, Any]]:
    """Healthy training for ``pre_steps``, then a grad-L2 spike that
    triggers exploding_grad and a grad-clip fix suggestion."""
    rng = random.Random(seed)
    base_ts = base_ts or time.time()
    out: list[dict[str, Any]] = [_run_started(run_id, base_ts)]
    step = 0

    # Healthy phase: loss decays, grad-L2 stable around 1.0.
    for i in range(pre_steps):
        ts = base_ts + (i + 1) * step_dt
        loss = 5.0 / (1.0 + i * 0.05) + rng.uniform(-0.05, 0.05)
        grad = 1.0 + rng.uniform(-0.15, 0.15)
        out.append(_step_event(run_id=run_id, step=i, ts=ts, loss=loss, grad_l2=grad))
        step = i

    # Spike phase: grad-L2 explodes, loss eventually NaNs.
    spike_step = step + 1
    spike_ts = base_ts + (pre_steps + 1) * step_dt
    for j in range(5):
        ts = spike_ts + j * step_dt
        grad = 50.0 + j * 30.0
        loss = 2.5 + j * 0.4
        out.append(_step_event(run_id=run_id, step=spike_step + j, ts=ts, loss=loss, grad_l2=grad))

    fire_step = spike_step + 4
    fire_ts = spike_ts + 4 * step_dt + step_dt / 2
    out.append(
        _anomaly(
            run_id=run_id,
            step=fire_step,
            ts=fire_ts,
            detector="exploding_grad",
            signal="grad_spike",
            severity="critical",
            evidence={"grad_l2": 170.0, "rolling_mean": 1.0, "ratio": 170.0},
        )
    )

    # Post-fix phase: grad-L2 settles after clip applied.
    for k in range(post_steps):
        ts = fire_ts + (k + 1) * step_dt
        grad = 0.95 + rng.uniform(-0.1, 0.1)
        loss = 1.5 / (1.0 + k * 0.05)
        out.append(
            _step_event(run_id=run_id, step=fire_step + 1 + k, ts=ts, loss=loss, grad_l2=grad)
        )

    out.append(_run_ended(run_id, ts + step_dt, fire_step + post_steps))
    return out


# ---- dead_relu ---------------------------------------------------------------


def generate_dead_relu(
    *,
    run_id: str = "demo-dead-relu",
    pre_steps: int = DEFAULT_PRE_STEPS,
    post_steps: int = DEFAULT_POST_STEPS,
    base_ts: float = 0.0,
    step_dt: float = 0.05,
    seed: int = 2,
) -> list[dict[str, Any]]:
    """Healthy training that progressively kills more ReLU units, then
    fires the dead_relu detector and proposes ReplaceActivation."""
    rng = random.Random(seed)
    base_ts = base_ts or time.time()
    out: list[dict[str, Any]] = [_run_started(run_id, base_ts)]

    # Phase 1: healthy ReLU activations.
    for i in range(pre_steps):
        ts = base_ts + (i + 1) * step_dt
        loss = 4.0 / (1.0 + i * 0.04) + rng.uniform(-0.05, 0.05)
        dead = 0.05 + rng.uniform(0.0, 0.1)
        acts = {
            "relu1": {"mean": 0.4, "dead_fraction": dead},
            "relu2": {"mean": 0.3, "dead_fraction": dead + 0.05},
        }
        out.append(
            _step_event(run_id=run_id, step=i, ts=ts, loss=loss, grad_l2=0.8, activations=acts)
        )

    # Phase 2: dead fraction climbs.
    bad_start = pre_steps
    for j in range(8):
        ts = base_ts + (bad_start + j + 1) * step_dt
        loss = 2.5 + rng.uniform(-0.1, 0.1)
        dead = 0.6 + j * 0.05
        acts = {
            "relu1": {"mean": 0.05, "dead_fraction": min(0.99, dead)},
            "relu2": {"mean": 0.03, "dead_fraction": min(0.99, dead + 0.1)},
        }
        out.append(
            _step_event(
                run_id=run_id,
                step=bad_start + j,
                ts=ts,
                loss=loss,
                grad_l2=0.4,
                activations=acts,
            )
        )

    fire_step = bad_start + 7
    fire_ts = base_ts + (bad_start + 8) * step_dt
    out.append(
        _anomaly(
            run_id=run_id,
            step=fire_step,
            ts=fire_ts,
            detector="dead_relu",
            signal="dead_fraction_high",
            severity="warning",
            evidence={"module": "relu2", "dead_fraction": 0.92},
        )
    )

    # Phase 3: post-fix recovery.
    for k in range(post_steps):
        ts = fire_ts + (k + 1) * step_dt
        loss = 2.0 / (1.0 + k * 0.04)
        acts = {
            "relu1": {"mean": 0.35, "dead_fraction": 0.1},
            "relu2": {"mean": 0.30, "dead_fraction": 0.12},
        }
        out.append(
            _step_event(
                run_id=run_id,
                step=fire_step + 1 + k,
                ts=ts,
                loss=loss,
                grad_l2=0.85,
                activations=acts,
            )
        )

    out.append(_run_ended(run_id, ts + step_dt, fire_step + post_steps))
    return out


# ---- missing_normalize -------------------------------------------------------


def generate_missing_normalize(
    *,
    run_id: str = "demo-missing-normalize",
    pre_steps: int = 5,
    post_steps: int = DEFAULT_POST_STEPS,
    base_ts: float = 0.0,
    step_dt: float = 0.05,
    seed: int = 3,
) -> list[dict[str, Any]]:
    """First batch publishes raw uint8 input_stats → fires
    missing_normalize on step 0 and proposes AddNormalize."""
    rng = random.Random(seed)
    base_ts = base_ts or time.time()
    out: list[dict[str, Any]] = [_run_started(run_id, base_ts)]

    # Step 0 — the trainer publishes input_stats once, at the start.
    ts0 = base_ts + step_dt
    out.append(
        _step_event(
            run_id=run_id,
            step=0,
            ts=ts0,
            loss=2.5,
            grad_l2=0.9,
            extra={
                "input_stats": {
                    "mean": 127.5,
                    "std": 60.0,
                    "min": 0.0,
                    "max": 255.0,
                }
            },
        )
    )
    out.append(
        _anomaly(
            run_id=run_id,
            step=0,
            ts=ts0 + step_dt / 2,
            detector="missing_normalize",
            signal="unnormalized_input",
            severity="warning",
            evidence={
                "input_mean": 127.5,
                "input_max": 255.0,
                "triggers": ["mean_out_of_range", "max_out_of_range", "unsigned_range"],
            },
        )
    )

    # A few stalled steps (loss not decreasing because the data is wrong).
    for i in range(1, pre_steps + 1):
        ts = base_ts + (i + 1) * step_dt
        loss = 2.4 + rng.uniform(-0.05, 0.05)  # plateaus
        out.append(_step_event(run_id=run_id, step=i, ts=ts, loss=loss, grad_l2=0.5))

    # Post-fix: normalized input → loss drops sharply, then settles.
    fire_step = pre_steps
    for k in range(post_steps):
        ts = base_ts + (pre_steps + 2 + k) * step_dt
        # Sharp recovery curve.
        loss = 0.8 + 1.5 * math.exp(-k * 0.15)
        out.append(
            _step_event(run_id=run_id, step=fire_step + 1 + k, ts=ts, loss=loss, grad_l2=0.9)
        )

    out.append(_run_ended(run_id, ts + step_dt, fire_step + post_steps))
    return out


SCENARIOS: dict[str, Any] = {
    "exploding_grad": generate_exploding_grad,
    "dead_relu": generate_dead_relu,
    "missing_normalize": generate_missing_normalize,
}


def list_scenarios() -> list[str]:
    return sorted(SCENARIOS.keys())


def generate(name: str, **kwargs: Any) -> list[dict[str, Any]]:
    fn = SCENARIOS.get(name)
    if fn is None:
        raise ValueError(f"unknown scenario {name!r}; choices: {', '.join(list_scenarios())}")
    return fn(**kwargs)


__all__ = [
    "SCENARIOS",
    "generate",
    "generate_dead_relu",
    "generate_exploding_grad",
    "generate_missing_normalize",
    "list_scenarios",
]
