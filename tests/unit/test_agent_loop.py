"""Unit tests for the diagnose-anomaly ReAct loop."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any
from unittest.mock import AsyncMock

import pytest

from lensml.agent.loop import MAX_CYCLES, diagnose_anomaly
from lensml.agent.tools import ToolContext
from lensml.llm.client import LLMResponse
from lensml.server.store import EventStore

# ---- helpers ----


def _resp(
    *,
    content: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    label: str = "primary",
) -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=tool_calls,
        model="fake/primary",
        label=label,
        latency_ms=1.0,
    )


def _tc(id_: str, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "id": id_,
        "name": name,
        "arguments": json.dumps(args or {}),
    }


def _scripted_llm(responses: Iterable[LLMResponse]) -> AsyncMock:
    """Return a mock LensmlLLM whose `.complete` yields scripted responses."""

    class _Llm:
        def __init__(self, items: list[LLMResponse]) -> None:
            self._iter = iter(items)
            self.complete = AsyncMock(side_effect=self._take)

        async def _take(self, *_args: Any, **_kwargs: Any) -> LLMResponse:
            try:
                return next(self._iter)
            except StopIteration as e:  # pragma: no cover — test-config error
                raise AssertionError("LLM called more times than scripted") from e

    return _Llm(list(responses))


@pytest.fixture
def tool_ctx() -> ToolContext:
    store = EventStore(":memory:")
    store.begin_run("r")
    # Seed one metric so tools have data to chew on.
    store.append_event("r", "metric", {"step": 0, "loss": 0.5, "grad_l2": 1.0})
    return ToolContext(store=store, run_id="r")


# ---- happy path ----


@pytest.mark.asyncio
async def test_happy_path_proposes_fix(tool_ctx: ToolContext) -> None:
    """Agent inspects evidence, then calls propose_fix with confidence=high."""
    llm = _scripted_llm(
        [
            # Cycle 0: inspect tools
            _resp(
                tool_calls=[
                    _tc("c1", "inspect_loss_curve", {"window": 50}),
                    _tc("c2", "inspect_gradient_flow", {"last_n_steps": 50}),
                ]
            ),
            # Cycle 1: propose
            _resp(
                tool_calls=[
                    _tc(
                        "c3",
                        "propose_fix",
                        {
                            "template": "AddGradientClipping",
                            "params": {"max_norm": 1.0},
                            "confidence": "high",
                            "rationale": "grad-L2 ratio > 10 over 3 steps",
                        },
                    )
                ]
            ),
        ]
    )

    state = await diagnose_anomaly(
        llm=llm,
        tool_ctx=tool_ctx,
        anomaly={"detector": "exploding_grad", "step": 12, "signal": "grad_spike"},
    )

    assert state.confidence == "high"
    assert len(state.proposed_fixes) == 1
    assert state.proposed_fixes[0]["template"] == "AddGradientClipping"
    assert state.cycles_used == 2

    # propose_fix records to store
    decisions = tool_ctx.store.list_decisions("r")
    assert len(decisions) == 1
    assert decisions[0]["fix_template"] == "AddGradientClipping"


@pytest.mark.asyncio
async def test_propose_fix_in_first_cycle(tool_ctx: ToolContext) -> None:
    """If the model goes straight to propose_fix, we still record it."""
    llm = _scripted_llm(
        [
            _resp(
                tool_calls=[
                    _tc(
                        "c1",
                        "propose_fix",
                        {
                            "template": "LowerLearningRate",
                            "params": {"factor": 0.3},
                            "confidence": "medium",
                        },
                    )
                ]
            )
        ]
    )
    state = await diagnose_anomaly(llm=llm, tool_ctx=tool_ctx, anomaly={"detector": "lr_too_high"})
    assert state.confidence == "medium"
    assert state.cycles_used == 1
    assert state.proposed_fixes[0]["template"] == "LowerLearningRate"


# ---- terminal text (no propose_fix) ----


@pytest.mark.asyncio
async def test_plain_text_response_is_terminal(tool_ctx: ToolContext) -> None:
    llm = _scripted_llm([_resp(content="I'm not sure what's wrong.")])
    state = await diagnose_anomaly(
        llm=llm, tool_ctx=tool_ctx, anomaly={"detector": "exploding_grad"}
    )
    assert state.confidence == "low"
    assert state.current_hypothesis is not None
    assert "not sure" in state.current_hypothesis
    assert state.cycles_used == 1
    assert state.proposed_fixes == []


# ---- cycle budget exhausted ----


@pytest.mark.asyncio
async def test_max_cycles_exhausted_returns_low(tool_ctx: ToolContext) -> None:
    """Agent calls a non-terminal tool every cycle without ever proposing."""
    spinning = [
        _resp(tool_calls=[_tc(f"c{i}", "inspect_loss_curve", {"window": 50})])
        for i in range(MAX_CYCLES)
    ]
    llm = _scripted_llm(spinning)

    state = await diagnose_anomaly(
        llm=llm, tool_ctx=tool_ctx, anomaly={"detector": "exploding_grad"}
    )
    assert state.confidence == "low"
    assert state.cycles_used == MAX_CYCLES
    assert state.proposed_fixes == []


# ---- robustness ----


@pytest.mark.asyncio
async def test_tool_error_is_reported_to_model_not_raised(
    tool_ctx: ToolContext,
) -> None:
    """A ToolError must not propagate — the loop captures and continues."""
    llm = _scripted_llm(
        [
            # cycle 0: bad tool name (raises ToolError)
            _resp(tool_calls=[_tc("c1", "does_not_exist", {})]),
            # cycle 1: sane proposal
            _resp(
                tool_calls=[
                    _tc(
                        "c2",
                        "propose_fix",
                        {
                            "template": "AddGradientClipping",
                            "params": {},
                            "confidence": "low",
                        },
                    )
                ]
            ),
        ]
    )
    state = await diagnose_anomaly(llm=llm, tool_ctx=tool_ctx, anomaly={"detector": "x"})
    # No exception escaped; loop continued past the bad tool.
    assert state.cycles_used == 2
    assert state.proposed_fixes[0]["template"] == "AddGradientClipping"


@pytest.mark.asyncio
async def test_llm_failure_returns_low_confidence(tool_ctx: ToolContext) -> None:
    """If LiteLLM raises (all routes failed), the loop must not crash."""

    class _BrokenLlm:
        complete = AsyncMock(side_effect=RuntimeError("all routes failed"))

    state = await diagnose_anomaly(
        llm=_BrokenLlm(),
        tool_ctx=tool_ctx,
        anomaly={"detector": "exploding_grad"},
    )
    assert state.confidence == "low"
    assert state.cycles_used == 1
    assert state.proposed_fixes == []


@pytest.mark.asyncio
async def test_propose_fix_default_confidence_when_unspecified(
    tool_ctx: ToolContext,
) -> None:
    """If LLM omits confidence, the loop falls back to medium (per plan §5.4
    — never below medium when a fix is proposed)."""
    llm = _scripted_llm(
        [
            _resp(
                tool_calls=[
                    _tc(
                        "c1",
                        "propose_fix",
                        # No 'confidence' field; tool's Pydantic schema
                        # defaults to "medium" — propose_fix still
                        # accepts it and the loop reads that back out.
                        {
                            "template": "X",
                            "params": {},
                            "rationale": "...",
                        },
                    )
                ]
            )
        ]
    )
    state = await diagnose_anomaly(llm=llm, tool_ctx=tool_ctx, anomaly={"detector": "x"})
    # The args dict that ended up on state.proposed_fixes is what the
    # LLM provided; the Pydantic schema default was applied during
    # tool execution but loop reads from the raw arg dict.
    # Since 'confidence' isn't in args, our get(...,"medium") kicks in.
    assert state.confidence == "medium"
