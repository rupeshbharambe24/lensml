"""Unit tests for lensml.agent.memory."""

from __future__ import annotations

import json

from lensml.agent.memory import (
    DEFAULT_VERBATIM_PAIRS,
    DiagnosisState,
    History,
    Observation,
    ToolCall,
)

# ---- DiagnosisState ----


def test_diagnosis_state_default_summary() -> None:
    s = DiagnosisState(run_id="r")
    assert "no hypotheses" in s.summary_for_prompt()


def test_diagnosis_state_summary_includes_fields() -> None:
    s = DiagnosisState(
        run_id="r",
        current_hypothesis="exploding gradient",
        confirmed=["grad_l2 spike"],
        ruled_out=["nan_loss"],
        proposed_fixes=[{"template": "AddGradientClipping"}],
    )
    out = s.summary_for_prompt()
    assert "exploding gradient" in out
    assert "grad_l2 spike" in out
    assert "nan_loss" in out
    assert "AddGradientClipping" in out


# ---- Observation ----


def test_observation_to_content_result() -> None:
    obs = Observation(tool_name="x", cycle=0, result={"a": 1})
    payload = json.loads(obs.to_content())
    assert payload == {"a": 1}
    assert obs.is_error is False


def test_observation_to_content_error() -> None:
    obs = Observation(tool_name="x", cycle=0, error="boom")
    payload = json.loads(obs.to_content())
    assert payload == {"error": "boom"}
    assert obs.is_error is True


# ---- History ----


def test_history_empty_renders_no_messages() -> None:
    h = History()
    assert h.to_messages() == []


def test_history_appends_pairs() -> None:
    h = History()
    h.append(
        ToolCall(id="t1", name="inspect_loss_curve", arguments={"window": 50}),
        Observation(tool_name="inspect_loss_curve", cycle=0, result={"slope": -0.01}),
    )
    assert len(h) == 1


def test_history_renders_assistant_and_tool_messages() -> None:
    h = History()
    h.append(
        ToolCall(id="t1", name="inspect_loss_curve", arguments={"window": 50}),
        Observation(tool_name="inspect_loss_curve", cycle=0, result={"slope": -0.01}),
    )
    msgs = h.to_messages()
    assert len(msgs) == 2
    assert msgs[0]["role"] == "assistant"
    assert msgs[0]["tool_calls"][0]["function"]["name"] == "inspect_loss_curve"
    assert msgs[1]["role"] == "tool"
    assert msgs[1]["tool_call_id"] == "t1"
    assert "slope" in msgs[1]["content"]


def test_history_masks_older_pairs_beyond_threshold() -> None:
    h = History(max_verbatim=3)
    for i in range(5):
        h.append(
            ToolCall(id=f"t{i}", name="inspect_loss_curve", arguments={"i": i}),
            Observation(tool_name="inspect_loss_curve", cycle=i, result={"i": i}),
        )

    msgs = h.to_messages()
    # 5 pairs * 2 messages per pair = 10 messages
    assert len(msgs) == 10
    # First 2 pairs (indices 0,1) should be masked since max_verbatim=3.
    masked_tool_msgs = [m for m in msgs[:4] if m["role"] == "tool"]
    assert len(masked_tool_msgs) == 2
    for m in masked_tool_msgs:
        body = json.loads(m["content"])
        assert body.get("masked") is True
    # Last 3 tool messages render their actual results.
    recent_tool_msgs = [m for m in msgs[4:] if m["role"] == "tool"]
    for m in recent_tool_msgs:
        body = json.loads(m["content"])
        assert body.get("masked") is not True


def test_history_default_max_verbatim() -> None:
    h = History()
    assert h.max_verbatim == DEFAULT_VERBATIM_PAIRS


def test_history_pairs_property_returns_copy() -> None:
    h = History()
    h.append(
        ToolCall(id="t1", name="x", arguments={}),
        Observation(tool_name="x", cycle=0, result={}),
    )
    pairs = h.pairs
    pairs.clear()  # mutating the copy must not affect the history
    assert len(h) == 1
