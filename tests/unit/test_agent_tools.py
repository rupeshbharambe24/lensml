"""Unit tests for the agent's tool registry (lensml.agent.tools)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from lensml.agent.tools import (
    REGISTRY,
    InspectActivationsInput,
    InspectDataloaderThroughputInput,
    InspectGradientFlowInput,
    InspectLossCurveInput,
    InspectLrScheduleInput,
    InspectOptimizerStateInput,
    InspectTrainValGapInput,
    ProposeFixInput,
    ReadTrainingScriptInput,
    ToolContext,
    ToolError,
    execute_tool,
    inspect_activations,
    inspect_dataloader_throughput,
    inspect_gradient_flow,
    inspect_loss_curve,
    inspect_lr_schedule,
    inspect_optimizer_state,
    inspect_train_val_gap,
    propose_fix,
    read_training_script,
    to_openai_tools,
)
from lensml.server.store import EventStore

# ---- fixtures ----


@pytest.fixture
def store() -> EventStore:
    s = EventStore(":memory:")
    yield s
    s.close()


@pytest.fixture
def ctx(store: EventStore) -> ToolContext:
    store.begin_run("r")
    return ToolContext(store=store, run_id="r")


def _seed_metrics(store: EventStore, run_id: str, payloads: list[dict]) -> None:
    """Append synthetic metric events with a given payload sequence."""
    for p in payloads:
        store.append_event(run_id, "metric", p)


# ---- registry shape ----


def test_registry_has_12_tools() -> None:
    assert len(REGISTRY) == 12


def test_to_openai_tools_format() -> None:
    tools = to_openai_tools()
    assert len(tools) == 12
    for t in tools:
        assert t["type"] == "function"
        fn = t["function"]
        assert "name" in fn
        assert "description" in fn
        assert "parameters" in fn
        assert fn["parameters"]["type"] == "object"


# ---- inspect_loss_curve ----


def test_inspect_loss_curve_empty(ctx: ToolContext) -> None:
    out = inspect_loss_curve(InspectLossCurveInput(), ctx)
    assert out["values"] == []
    assert out["n"] == 0


def test_inspect_loss_curve_basic(ctx: ToolContext, store: EventStore) -> None:
    losses = [1.0, 0.9, 0.8, 0.7, 0.6]
    _seed_metrics(store, "r", [{"step": i, "loss": v} for i, v in enumerate(losses)])
    out = inspect_loss_curve(InspectLossCurveInput(window=10), ctx)
    assert out["n"] == 5
    assert out["values"] == losses
    assert out["slope"] < 0  # descending
    assert out["nan_count"] == 0


def test_inspect_loss_curve_counts_nan(ctx: ToolContext, store: EventStore) -> None:
    _seed_metrics(
        store,
        "r",
        [
            {"step": 0, "loss": 1.0},
            {"step": 1, "loss": float("nan")},
            {"step": 2, "loss": float("nan")},
        ],
    )
    out = inspect_loss_curve(InspectLossCurveInput(), ctx)
    assert out["nan_count"] == 2


def test_inspect_loss_curve_caps_returned_values(ctx: ToolContext, store: EventStore) -> None:
    _seed_metrics(store, "r", [{"step": i, "loss": 1.0} for i in range(200)])
    out = inspect_loss_curve(InspectLossCurveInput(window=200), ctx)
    assert len(out["values"]) <= 50


# ---- inspect_gradient_flow ----


def test_inspect_gradient_flow_basic(ctx: ToolContext, store: EventStore) -> None:
    _seed_metrics(store, "r", [{"step": i, "grad_l2": 1.0 + i * 0.1} for i in range(10)])
    out = inspect_gradient_flow(InspectGradientFlowInput(), ctx)
    g = out["global_grad_l2"]
    assert g["n"] == 10
    assert g["min"] == pytest.approx(1.0)
    assert g["max"] == pytest.approx(1.9)
    assert out["trend"] > 0


def test_inspect_gradient_flow_empty(ctx: ToolContext) -> None:
    out = inspect_gradient_flow(InspectGradientFlowInput(), ctx)
    assert out["global_grad_l2"]["n"] == 0


# ---- inspect_activations ----


def test_inspect_activations_aggregates_per_layer(ctx: ToolContext, store: EventStore) -> None:
    _seed_metrics(
        store,
        "r",
        [
            {
                "step": i,
                "activations": {
                    "relu1": {"mean": 0.5, "dead_fraction": 0.5},
                    "relu2": {"mean": 0.1, "dead_fraction": 0.9},
                },
            }
            for i in range(5)
        ],
    )
    out = inspect_activations(InspectActivationsInput(), ctx)
    assert out["n_layers"] == 2
    assert out["layers"]["relu1"]["mean"] == pytest.approx(0.5)
    assert out["layers"]["relu2"]["dead_fraction_mean"] == pytest.approx(0.9)


def test_inspect_activations_layer_pattern_filter(ctx: ToolContext, store: EventStore) -> None:
    _seed_metrics(
        store,
        "r",
        [
            {
                "activations": {
                    "relu1": {"mean": 0.1, "dead_fraction": 0.0},
                    "fc.weight": {"mean": 0.2, "dead_fraction": 0.0},
                }
            }
        ],
    )
    out = inspect_activations(InspectActivationsInput(layer_pattern="relu"), ctx)
    assert set(out["layers"].keys()) == {"relu1"}


# ---- inspect_optimizer_state / inspect_lr_schedule ----


def test_inspect_optimizer_state(ctx: ToolContext, store: EventStore) -> None:
    _seed_metrics(
        store,
        "r",
        [
            {"step": 0, "lr": 0.01, "ts": 1.0},
            {"step": 1, "lr": 0.01, "ts": 2.0},
        ],
    )
    out = inspect_optimizer_state(InspectOptimizerStateInput(), ctx)
    assert out["lr"] == pytest.approx(0.01)
    assert out["step"] == 1


def test_inspect_optimizer_state_empty(ctx: ToolContext) -> None:
    out = inspect_optimizer_state(InspectOptimizerStateInput(), ctx)
    assert out["available"] is False


def test_inspect_lr_schedule_constant(ctx: ToolContext, store: EventStore) -> None:
    _seed_metrics(store, "r", [{"step": i, "lr": 0.01} for i in range(10)])
    out = inspect_lr_schedule(InspectLrScheduleInput(), ctx)
    assert out["schedule"] == "constant"
    assert out["current_lr"] == pytest.approx(0.01)


def test_inspect_lr_schedule_varying(ctx: ToolContext, store: EventStore) -> None:
    _seed_metrics(store, "r", [{"step": i, "lr": 0.01 * (0.9**i)} for i in range(10)])
    out = inspect_lr_schedule(InspectLrScheduleInput(), ctx)
    assert out["schedule"] == "varying"
    assert out["trend"] < 0


# ---- inspect_train_val_gap ----


def test_inspect_train_val_gap_with_logged_val(ctx: ToolContext, store: EventStore) -> None:
    _seed_metrics(
        store,
        "r",
        [
            {"step": 0, "loss": 0.5, "val_loss": 0.7},
            {"step": 1, "loss": 0.4, "val_loss": 0.6},
        ],
    )
    out = inspect_train_val_gap(InspectTrainValGapInput(metric="loss"), ctx)
    assert out["train_latest"] == pytest.approx(0.4)
    assert out["val_latest"] == pytest.approx(0.6)
    assert out["gap"] == pytest.approx(0.2)


def test_inspect_train_val_gap_no_val(ctx: ToolContext, store: EventStore) -> None:
    _seed_metrics(store, "r", [{"step": 0, "loss": 0.5}])
    out = inspect_train_val_gap(InspectTrainValGapInput(metric="loss"), ctx)
    assert out["available"] is False


# ---- inspect_dataloader_throughput ----


def test_inspect_dataloader_throughput(ctx: ToolContext, store: EventStore) -> None:
    base = time.time()
    _seed_metrics(store, "r", [{"step": i, "ts": base + i * 0.1} for i in range(10)])
    out = inspect_dataloader_throughput(InspectDataloaderThroughputInput(), ctx)
    # 10 events spanning 0.9 s → ~10 batches/sec
    assert out["batches_per_sec"] > 5.0
    assert out["batches_per_sec"] < 20.0


def test_inspect_dataloader_throughput_insufficient_data(
    ctx: ToolContext, store: EventStore
) -> None:
    _seed_metrics(store, "r", [{"step": 0, "ts": 1.0}])
    out = inspect_dataloader_throughput(InspectDataloaderThroughputInput(), ctx)
    assert out["batches_per_sec"] == 0.0


# ---- read_training_script ----


def test_read_training_script_returns_slice(ctx: ToolContext, tmp_path: Path) -> None:
    script = tmp_path / "train.py"
    script.write_text("\n".join(f"# line {i}" for i in range(1, 21)))
    ctx.script_path = script

    out = read_training_script(ReadTrainingScriptInput(start_line=5, end_line=8), ctx)
    assert out["available"] is True
    assert "line 5" in out["text"]
    assert "line 8" in out["text"]
    assert "line 4" not in out["text"]
    assert "line 9" not in out["text"]
    assert out["total_lines"] == 20


def test_read_training_script_no_path(ctx: ToolContext) -> None:
    out = read_training_script(ReadTrainingScriptInput(), ctx)
    assert out["available"] is False


def test_read_training_script_missing_file(ctx: ToolContext, tmp_path: Path) -> None:
    ctx.script_path = tmp_path / "nope.py"
    out = read_training_script(ReadTrainingScriptInput(), ctx)
    assert out["available"] is False


# ---- propose_fix ----


def test_propose_fix_records_decision(ctx: ToolContext, store: EventStore) -> None:
    out = propose_fix(
        ProposeFixInput(
            template="AddGradientClipping",
            params={"max_norm": 1.0},
            confidence="high",
            rationale="Grad-L2 spiked.",
        ),
        ctx,
    )
    assert out["queued"] is True
    assert out["template"] == "AddGradientClipping"
    decisions = store.list_decisions("r")
    assert len(decisions) == 1
    assert decisions[0]["fix_template"] == "AddGradientClipping"


def test_propose_fix_invalid_confidence_raises(ctx: ToolContext) -> None:
    with pytest.raises(ToolError):
        propose_fix(
            ProposeFixInput(template="x", confidence="urgent"),
            ctx,
        )


# ---- execute_tool dispatcher ----


def test_execute_tool_dispatches(ctx: ToolContext, store: EventStore) -> None:
    _seed_metrics(store, "r", [{"step": 0, "loss": 1.0}])
    out = execute_tool("inspect_loss_curve", {}, ctx)
    assert out["n"] == 1


def test_execute_tool_accepts_json_string_args(ctx: ToolContext, store: EventStore) -> None:
    _seed_metrics(store, "r", [{"step": i, "loss": 1.0} for i in range(5)])
    out = execute_tool("inspect_loss_curve", '{"window": 3}', ctx)
    assert out["n"] >= 1


def test_execute_tool_unknown_raises(ctx: ToolContext) -> None:
    with pytest.raises(ToolError, match="unknown tool"):
        execute_tool("does_not_exist", {}, ctx)


def test_execute_tool_invalid_args_raises(ctx: ToolContext) -> None:
    # window must be >= 1
    with pytest.raises(ToolError, match="invalid arguments"):
        execute_tool("inspect_loss_curve", {"window": 0}, ctx)


def test_execute_tool_invalid_json_string_raises(ctx: ToolContext) -> None:
    with pytest.raises(ToolError, match="invalid JSON"):
        execute_tool("inspect_loss_curve", "not json", ctx)


def test_execute_tool_empty_json_string_treated_as_empty_dict(
    ctx: ToolContext, store: EventStore
) -> None:
    _seed_metrics(store, "r", [{"step": 0, "loss": 1.0}])
    out = execute_tool("inspect_loss_curve", "", ctx)
    assert "values" in out


def test_execute_tool_propose_fix_dispatch(ctx: ToolContext, store: EventStore) -> None:
    out = execute_tool(
        "propose_fix",
        {"template": "X", "params": {}, "confidence": "low", "rationale": "test"},
        ctx,
    )
    assert out["queued"] is True
