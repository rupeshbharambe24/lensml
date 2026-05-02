"""The agent's diagnostic vocabulary — 12 tools per plan §5.2.

Each tool:
    - Has a Pydantic input schema (used both for validation and to produce
      the JSON Schema fed to the LLM via LiteLLM).
    - Returns a small dict (target ≤2KB per plan §5.2 — keeps the agent's
      context budget bounded across many tool calls).
    - Is read-only EXCEPT `propose_fix`, which queues an ApprovalRequest
      for the user to approve via the dashboard. The actual git+libcst
      application happens in step 2.5; this tool just records intent.

Server-side execution: the agent emits a tool_call → the server's runtime
reads the args, validates, dispatches to the function, and returns the
dict back to the agent. ToolContext bundles the references each tool may
need (event store, run_id, optional script path / model handle).

Many tools currently read from the persisted event log (lensml.server.store)
populated by the `lensml.watch()` integration. Tools that need data we
don't capture yet (data batches, GPU memory, dataloader throughput) return
a structured "unavailable" response and are flagged in their description
so the agent knows to skip them. Those become real in v0.2.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from lensml.server.store import EventStore


# ----------------------------------------------------------------------------
# Errors and shared types
# ----------------------------------------------------------------------------


class ToolError(Exception):
    """Raised when a tool can't be dispatched / validated / executed."""


@dataclass
class ToolContext:
    """Per-call context the runtime gives to each tool's execute fn.

    `store` and `run_id` are always present. Optional fields (script_path,
    model, optimizer) are populated when the trainer or session bootstrap
    wired them in; tools that need them but don't have them return an
    "unavailable" response.
    """

    store: EventStore
    run_id: str
    script_path: Path | None = None
    # Future: model + optimizer references for in-process inspection.


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


_MAX_VALUES_RETURNED = 50  # cap series-shaped responses (plan §5.2 ≤2KB)


def _metric_payloads(ctx: ToolContext, *, limit: int = 1000) -> list[dict[str, Any]]:
    """Return the payload dicts of recent metric events for the run."""
    events = ctx.store.get_events(ctx.run_id, kind="metric", limit=limit)
    return [e.get("payload", {}) for e in events]


def _linear_slope(values: list[float]) -> float:
    """Least-squares slope of `values` (treating index as x). 0.0 for n<2."""
    n = len(values)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(values) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, values, strict=True))
    den = sum((x - mean_x) ** 2 for x in xs)
    return float(num / den) if den != 0 else 0.0


def _plateau_score(values: list[float]) -> float:
    """0..1 score: 1 == fully flat (no progress), 0 == strong descent.

    Defined as the fraction of step-to-step deltas with |delta| < tolerance.
    """
    if len(values) < 2:
        return 0.0
    deltas = [abs(values[i] - values[i - 1]) for i in range(1, len(values))]
    median_v = statistics.median([abs(v) for v in values]) or 1.0
    tol = median_v * 0.005
    flat = sum(1 for d in deltas if d < tol)
    return flat / len(deltas)


def _rolling_smooth(values: list[float], window: int = 5) -> list[float]:
    if window < 2 or len(values) < window:
        return values
    out: list[float] = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        out.append(sum(values[lo : i + 1]) / (i - lo + 1))
    return out


# ============================================================================
# 1. inspect_loss_curve
# ============================================================================


class InspectLossCurveInput(BaseModel):
    window: int = Field(
        default=200, ge=1, le=10000, description="How many recent steps to consider."
    )
    smooth: bool = Field(
        default=False, description="Apply rolling-mean smoothing to returned values."
    )


def inspect_loss_curve(args: InspectLossCurveInput, ctx: ToolContext) -> dict[str, Any]:
    payloads = _metric_payloads(ctx, limit=args.window)
    losses: list[float] = []
    for p in payloads:
        v = p.get("loss")
        if isinstance(v, (int, float)):
            losses.append(float(v))

    if not losses:
        return {
            "values": [],
            "slope": 0.0,
            "plateau_score": 0.0,
            "nan_count": 0,
            "std": 0.0,
            "n": 0,
        }

    nan_count = sum(1 for v in losses if math.isnan(v))
    finite = [v for v in losses if math.isfinite(v)]
    series = _rolling_smooth(finite, 5) if args.smooth else finite

    return {
        "values": series[-_MAX_VALUES_RETURNED:],
        "slope": _linear_slope(finite),
        "plateau_score": _plateau_score(finite),
        "nan_count": nan_count,
        "std": float(statistics.stdev(finite)) if len(finite) > 1 else 0.0,
        "n": len(losses),
    }


# ============================================================================
# 2. inspect_gradient_flow
# ============================================================================


class InspectGradientFlowInput(BaseModel):
    last_n_steps: int = Field(default=50, ge=1, le=10000, description="Recent step window.")


def inspect_gradient_flow(args: InspectGradientFlowInput, ctx: ToolContext) -> dict[str, Any]:
    payloads = _metric_payloads(ctx, limit=args.last_n_steps)
    grads: list[float] = []
    for p in payloads:
        v = p.get("grad_l2")
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            grads.append(float(v))

    if not grads:
        return {
            "global_grad_l2": {"mean": 0.0, "max": 0.0, "min": 0.0, "n": 0},
            "trend": 0.0,
        }

    return {
        "global_grad_l2": {
            "mean": sum(grads) / len(grads),
            "max": max(grads),
            "min": min(grads),
            "n": len(grads),
        },
        "trend": _linear_slope(grads),
    }


# ============================================================================
# 3. inspect_activations
# ============================================================================


class InspectActivationsInput(BaseModel):
    layer_pattern: str | None = Field(
        default=None,
        description="Substring filter on layer names. None returns all hooked layers.",
    )
    last_n_steps: int = Field(default=20, ge=1, le=1000)


def inspect_activations(args: InspectActivationsInput, ctx: ToolContext) -> dict[str, Any]:
    payloads = _metric_payloads(ctx, limit=args.last_n_steps)
    # Aggregate: layer -> list of {mean, dead_fraction}
    per_layer: dict[str, list[dict[str, float]]] = {}
    for p in payloads:
        acts = p.get("activations")
        if not isinstance(acts, dict):
            continue
        for name, stats in acts.items():
            if args.layer_pattern and args.layer_pattern not in name:
                continue
            if not isinstance(stats, dict):
                continue
            per_layer.setdefault(name, []).append(
                {
                    "mean": float(stats.get("mean", 0.0)),
                    "dead_fraction": float(stats.get("dead_fraction", 0.0)),
                }
            )

    summary: dict[str, dict[str, float]] = {}
    for name, samples in per_layer.items():
        if not samples:
            continue
        means = [s["mean"] for s in samples]
        deads = [s["dead_fraction"] for s in samples]
        summary[name] = {
            "mean": sum(means) / len(means),
            "mean_max": max(means),
            "dead_fraction_mean": sum(deads) / len(deads),
            "dead_fraction_max": max(deads),
            "n_samples": len(samples),
        }
    return {"layers": summary, "n_layers": len(summary)}


# ============================================================================
# 4. inspect_data_batch (v0.2 — placeholder)
# ============================================================================


class InspectDataBatchInput(BaseModel):
    n: int = Field(default=1, ge=1, le=8, description="How many recent batches to summarize.")


def inspect_data_batch(args: InspectDataBatchInput, ctx: ToolContext) -> dict[str, Any]:
    return {
        "available": False,
        "reason": (
            "Data-batch inspection requires trainer-side instrumentation that "
            "captures recent input tensors. Lands in v0.2."
        ),
    }


# ============================================================================
# 5. inspect_dataloader_throughput (partial — uses event timestamps)
# ============================================================================


class InspectDataloaderThroughputInput(BaseModel):
    window: int = Field(default=100, ge=2, le=10000)


def inspect_dataloader_throughput(
    args: InspectDataloaderThroughputInput, ctx: ToolContext
) -> dict[str, Any]:
    events = ctx.store.get_events(ctx.run_id, kind="metric", limit=args.window)
    # Prefer the trainer-published ts inside the payload (closer to when
    # the step actually happened); fall back to the store's append ts.
    timestamps: list[float] = []
    for e in events:
        payload = e.get("payload", {})
        ts = payload.get("ts") if isinstance(payload, dict) else None
        if not isinstance(ts, (int, float)):
            ts = e.get("ts")
        if isinstance(ts, (int, float)):
            timestamps.append(float(ts))
    if len(timestamps) < 2:
        return {"batches_per_sec": 0.0, "n_samples": len(timestamps)}
    span = timestamps[-1] - timestamps[0]
    if span <= 0:
        return {"batches_per_sec": 0.0, "n_samples": len(timestamps)}
    return {
        "batches_per_sec": (len(timestamps) - 1) / span,
        "n_samples": len(timestamps),
        "gpu_idle_fraction": None,  # v0.2: requires CUDA event timing
        "worker_count": None,  # v0.2: requires DataLoader instrumentation
    }


# ============================================================================
# 6. inspect_optimizer_state
# ============================================================================


class InspectOptimizerStateInput(BaseModel):
    pass


def inspect_optimizer_state(_args: InspectOptimizerStateInput, ctx: ToolContext) -> dict[str, Any]:
    payloads = _metric_payloads(ctx, limit=10)
    if not payloads:
        return {"available": False, "reason": "no metric events yet"}
    latest = payloads[-1]
    return {
        "lr": latest.get("lr"),
        "step": latest.get("step"),
        "ts": latest.get("ts"),
    }


# ============================================================================
# 7. inspect_lr_schedule
# ============================================================================


class InspectLrScheduleInput(BaseModel):
    last_n_steps: int = Field(default=200, ge=2, le=10000)


def inspect_lr_schedule(args: InspectLrScheduleInput, ctx: ToolContext) -> dict[str, Any]:
    payloads = _metric_payloads(ctx, limit=args.last_n_steps)
    lrs: list[float] = []
    for p in payloads:
        v = p.get("lr")
        if isinstance(v, (int, float)):
            lrs.append(float(v))
    if not lrs:
        return {"current_lr": None, "schedule": "unknown", "trend": 0.0}
    unique = len(set(round(v, 8) for v in lrs))
    schedule_kind = "constant" if unique == 1 else "varying"
    return {
        "current_lr": lrs[-1],
        "schedule": schedule_kind,
        "trend": _linear_slope(lrs),
        "n": len(lrs),
    }


# ============================================================================
# 8. inspect_model_summary (v0.2 — placeholder)
# ============================================================================


class InspectModelSummaryInput(BaseModel):
    pass


def inspect_model_summary(_args: InspectModelSummaryInput, ctx: ToolContext) -> dict[str, Any]:
    # Best-effort: derive layer names from the most recent activations dict.
    payloads = _metric_payloads(ctx, limit=5)
    layer_names: list[str] = []
    for p in payloads:
        acts = p.get("activations")
        if isinstance(acts, dict):
            layer_names = list(acts.keys())
            break
    return {
        "available": False,
        "reason": (
            "Full model summary (param_count, layer_types) requires trainer-side "
            "introspection. Lands in v0.2."
        ),
        "hooked_activation_layers": layer_names,
    }


# ============================================================================
# 9. inspect_train_val_gap
# ============================================================================


class InspectTrainValGapInput(BaseModel):
    metric: str = Field(
        default="loss",
        description="Metric name in the user's `handle.log()` calls (e.g. 'loss', 'accuracy').",
    )
    val_metric: str | None = Field(
        default=None,
        description="Validation metric name (default: 'val_<metric>').",
    )


def inspect_train_val_gap(args: InspectTrainValGapInput, ctx: ToolContext) -> dict[str, Any]:
    payloads = _metric_payloads(ctx, limit=500)
    train_key = args.metric
    val_key = args.val_metric or f"val_{args.metric}"

    train_vals: list[float] = []
    val_vals: list[float] = []
    for p in payloads:
        if isinstance(p.get(train_key), (int, float)):
            train_vals.append(float(p[train_key]))
        if isinstance(p.get(val_key), (int, float)):
            val_vals.append(float(p[val_key]))

    if not val_vals:
        return {
            "train_metric": train_key,
            "val_metric": val_key,
            "train_latest": train_vals[-1] if train_vals else None,
            "val_latest": None,
            "gap": None,
            "available": False,
            "reason": (
                f"No '{val_key}' values logged. Use `handle.log({val_key}=...)` from your "
                "validation loop."
            ),
        }
    return {
        "train_metric": train_key,
        "val_metric": val_key,
        "train_latest": train_vals[-1] if train_vals else None,
        "val_latest": val_vals[-1],
        "gap": (val_vals[-1] - train_vals[-1]) if train_vals else None,
    }


# ============================================================================
# 10. inspect_memory (v0.2 — placeholder)
# ============================================================================


class InspectMemoryInput(BaseModel):
    pass


def inspect_memory(_args: InspectMemoryInput, ctx: ToolContext) -> dict[str, Any]:
    return {
        "available": False,
        "reason": (
            "GPU memory introspection requires trainer-side `torch.cuda.memory_*` "
            "snapshots published in metric events. Lands in v0.2."
        ),
    }


# ============================================================================
# 11. read_training_script
# ============================================================================


class ReadTrainingScriptInput(BaseModel):
    start_line: int = Field(default=1, ge=1, description="1-based start line, inclusive.")
    end_line: int = Field(default=200, ge=1, description="1-based end line, inclusive.")


_MAX_SCRIPT_BYTES = 16_384  # bound the per-call response


def read_training_script(args: ReadTrainingScriptInput, ctx: ToolContext) -> dict[str, Any]:
    if ctx.script_path is None:
        return {
            "available": False,
            "reason": "No script_path bound to this tool context.",
        }
    if not ctx.script_path.exists():
        return {
            "available": False,
            "reason": f"Script not found at {ctx.script_path}",
        }
    if args.end_line < args.start_line:
        return {"available": False, "reason": "end_line < start_line"}

    lines = ctx.script_path.read_text(encoding="utf-8").splitlines()
    snippet_lines = lines[args.start_line - 1 : args.end_line]
    text = "\n".join(snippet_lines)
    truncated = len(text) > _MAX_SCRIPT_BYTES
    if truncated:
        text = text[:_MAX_SCRIPT_BYTES]
    return {
        "available": True,
        "path": str(ctx.script_path),
        "start_line": args.start_line,
        "end_line": args.start_line + len(snippet_lines) - 1,
        "total_lines": len(lines),
        "text": text,
        "truncated": truncated,
    }


# ============================================================================
# 12. propose_fix (write-side)
# ============================================================================


class ProposeFixInput(BaseModel):
    template: str = Field(description="Fix-template slug (e.g. 'AddGradientClipping').")
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Template-specific parameters (e.g. {'max_norm': 1.0}).",
    )
    confidence: str = Field(
        default="medium",
        description="One of: 'low', 'medium', 'high' (per plan §5.4 self-consistency).",
    )
    rationale: str = Field(
        default="",
        description="Free-text explanation shown to the user alongside the diff.",
    )


def propose_fix(args: ProposeFixInput, ctx: ToolContext) -> dict[str, Any]:
    """Queue a fix for user approval.

    Step 2.2 records the proposal in the `decisions` table; step 2.5 wires
    this to the dashboard's diff viewer + libcst application + git commit.
    """
    if args.confidence not in ("low", "medium", "high"):
        raise ToolError(f"invalid confidence: {args.confidence!r}")

    diff_text = json.dumps(
        {"template": args.template, "params": args.params, "rationale": args.rationale},
        sort_keys=True,
    )
    decision_id = ctx.store.record_decision(
        ctx.run_id,
        fix_template=args.template,
        diff_text=diff_text,
    )
    return {
        "queued": True,
        "decision_id": decision_id,
        "template": args.template,
        "confidence": args.confidence,
    }


# ----------------------------------------------------------------------------
# Registry + runtime
# ----------------------------------------------------------------------------


@dataclass
class Tool:
    name: str
    description: str
    input_schema: type[BaseModel]
    execute: Callable[[Any, ToolContext], dict[str, Any]]


REGISTRY: dict[str, Tool] = {
    "inspect_loss_curve": Tool(
        name="inspect_loss_curve",
        description=(
            "Summarize the recent training loss curve: last N values, "
            "least-squares slope, plateau score (0..1), NaN count, std."
        ),
        input_schema=InspectLossCurveInput,
        execute=inspect_loss_curve,
    ),
    "inspect_gradient_flow": Tool(
        name="inspect_gradient_flow",
        description=(
            "Report global grad-L2 statistics over the recent N steps: "
            "mean, max, min, and trend (slope)."
        ),
        input_schema=InspectGradientFlowInput,
        execute=inspect_gradient_flow,
    ),
    "inspect_activations": Tool(
        name="inspect_activations",
        description=(
            "Per-activation-layer mean and dead_fraction summarized over the "
            "last N steps. Optional layer_pattern is a substring filter."
        ),
        input_schema=InspectActivationsInput,
        execute=inspect_activations,
    ),
    "inspect_data_batch": Tool(
        name="inspect_data_batch",
        description=(
            "Summarize recent input batches (shapes, dtypes, label distribution, "
            "value range). v0.2 — currently returns 'unavailable'."
        ),
        input_schema=InspectDataBatchInput,
        execute=inspect_data_batch,
    ),
    "inspect_dataloader_throughput": Tool(
        name="inspect_dataloader_throughput",
        description=(
            "Estimate dataloader throughput (batches/sec) from event timestamps. "
            "GPU idle fraction + worker count are v0.2."
        ),
        input_schema=InspectDataloaderThroughputInput,
        execute=inspect_dataloader_throughput,
    ),
    "inspect_optimizer_state": Tool(
        name="inspect_optimizer_state",
        description="Current learning rate, step counter, and last update timestamp.",
        input_schema=InspectOptimizerStateInput,
        execute=inspect_optimizer_state,
    ),
    "inspect_lr_schedule": Tool(
        name="inspect_lr_schedule",
        description=(
            "Detect whether lr is constant or varying over the last N steps; "
            "returns current_lr and trend."
        ),
        input_schema=InspectLrScheduleInput,
        execute=inspect_lr_schedule,
    ),
    "inspect_model_summary": Tool(
        name="inspect_model_summary",
        description=("Hooked activation layer names and (v0.2) full model topology / param count."),
        input_schema=InspectModelSummaryInput,
        execute=inspect_model_summary,
    ),
    "inspect_train_val_gap": Tool(
        name="inspect_train_val_gap",
        description=(
            "Latest train metric vs latest val metric and the gap between them. "
            "Requires the user to call handle.log(val_<metric>=...) from their "
            "validation loop."
        ),
        input_schema=InspectTrainValGapInput,
        execute=inspect_train_val_gap,
    ),
    "inspect_memory": Tool(
        name="inspect_memory",
        description="GPU memory snapshot. v0.2 — currently returns 'unavailable'.",
        input_schema=InspectMemoryInput,
        execute=inspect_memory,
    ),
    "read_training_script": Tool(
        name="read_training_script",
        description=(
            "Return a slice of the user's training script source (start_line .. "
            "end_line, 1-based). Bounded at 16 KB."
        ),
        input_schema=ReadTrainingScriptInput,
        execute=read_training_script,
    ),
    "propose_fix": Tool(
        name="propose_fix",
        description=(
            "Queue a fix for user approval. Use a fix-template slug (e.g. "
            "'AddGradientClipping'), provide template-specific params, set "
            "confidence (low|medium|high), and explain in `rationale`. The "
            "actual code change is NOT applied until the user approves."
        ),
        input_schema=ProposeFixInput,
        execute=propose_fix,
    ),
}


def to_openai_tools() -> list[dict[str, Any]]:
    """Export the registry as OpenAI-format tool definitions for LiteLLM."""
    out: list[dict[str, Any]] = []
    for tool in REGISTRY.values():
        out.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema.model_json_schema(),
                },
            }
        )
    return out


def execute_tool(name: str, arguments: dict[str, Any] | str, ctx: ToolContext) -> dict[str, Any]:
    """Validate args + dispatch to the named tool. Raises `ToolError` on
    any unrecoverable problem."""
    tool = REGISTRY.get(name)
    if tool is None:
        raise ToolError(f"unknown tool: {name!r}")

    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError as e:
            raise ToolError(f"invalid JSON arguments for {name}: {e}") from e

    if not isinstance(arguments, dict):
        raise ToolError(f"arguments for {name} must be a dict, got {type(arguments).__name__}")

    try:
        validated = tool.input_schema.model_validate(arguments)
    except Exception as e:
        raise ToolError(f"invalid arguments for {name}: {e}") from e

    return tool.execute(validated, ctx)
