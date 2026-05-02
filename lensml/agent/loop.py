"""ReAct-style diagnose-anomaly loop — plan §5.1.

Single-agent, hand-rolled (~120 LOC). No LangChain / LangGraph / Pydantic-AI.
Reasons (plan §5.1):
    1. Every framework adds debug surface area.
    2. Solo + 30-day budget = control matters more than abstraction.
    3. The loop is genuinely small.

Flow (one anomaly):
    1. Build initial messages: system prompt + anomaly description.
    2. For up to MAX_CYCLES iterations:
        - Call LiteLLM with the full message list + tool definitions.
        - If the model returned tool calls: execute each, append the
          assistant + tool messages to the conversation, continue.
        - If `propose_fix` was called: record it on the state, return.
        - If the model returned plain text (no tool calls): treat as
          a terminal text answer, return with confidence="low".
    3. If we hit MAX_CYCLES without a fix proposal: return low confidence.

Tool errors do NOT raise — they're serialized into the tool message so
the model can see what went wrong and try a different tool.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from lensml.agent.memory import DiagnosisState, History, Observation, ToolCall
from lensml.agent.tools import ToolError, execute_tool, to_openai_tools
from lensml.llm.routing import Task

if TYPE_CHECKING:
    from lensml.agent.tools import ToolContext
    from lensml.llm.client import LensmlLLM


log = logging.getLogger(__name__)

MAX_CYCLES = 8
SYSTEM_PROMPT_TEMPLATE = """You are lensml, a senior ML engineer diagnosing a training-run pathology.

You have a budget of {max_cycles} cycles. Each cycle you may call one or
more diagnostic tools (read-only). When you have enough evidence, call
the `propose_fix` tool with a fix template, parameters, confidence
(low|medium|high), and a 1-2 sentence rationale.

Hard rules:
- Always cite specific evidence (numbers from tool results) in the rationale.
- Never propose with confidence="high" unless evidence is unambiguous.
- If unsure after 5 cycles, propose with confidence="low" and stop.

Current diagnosis state:
{state_summary}
"""


def _build_system_message(state: DiagnosisState) -> dict[str, str]:
    return {
        "role": "system",
        "content": SYSTEM_PROMPT_TEMPLATE.format(
            max_cycles=MAX_CYCLES,
            state_summary=state.summary_for_prompt(),
        ),
    }


def _build_user_message(anomaly: dict[str, Any]) -> dict[str, str]:
    return {
        "role": "user",
        "content": (
            "An anomaly was detected during training:\n\n"
            f"{json.dumps(anomaly, indent=2, default=str)}\n\n"
            "Investigate the root cause using the available tools, then propose a fix."
        ),
    }


def _parse_tool_args(arguments: str | dict[str, Any]) -> dict[str, Any]:
    """LiteLLM hands us either a dict or a JSON string — normalize."""
    if isinstance(arguments, dict):
        return arguments
    if not arguments:
        return {}
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


async def diagnose_anomaly(
    *,
    llm: LensmlLLM,
    tool_ctx: ToolContext,
    anomaly: dict[str, Any],
    max_cycles: int = MAX_CYCLES,
) -> DiagnosisState:
    """Run the ReAct loop for a single anomaly.

    Returns a populated `DiagnosisState`. The agent's `propose_fix` tool
    call (if any) is recorded on the state's `proposed_fixes` list AND
    persisted to the event store via the tool runtime.
    """
    state = DiagnosisState(run_id=tool_ctx.run_id, anomaly=anomaly)
    history = History()
    tools = to_openai_tools()

    for cycle in range(max_cycles):
        state.cycles_used = cycle + 1
        messages = [_build_system_message(state), _build_user_message(anomaly)]
        messages.extend(history.to_messages())

        try:
            response = await llm.complete(
                Task.DIAGNOSIS,
                messages,
                tools=tools,
                temperature=0.0,
            )
        except Exception:
            log.exception("LLM call failed in diagnose_anomaly cycle %d", cycle)
            state.confidence = "low"
            return state

        if not response.tool_calls:
            # Plain-text terminal answer.
            if response.content:
                state.current_hypothesis = response.content[:500]
            state.confidence = "low"
            return state

        # Run every tool the model asked for in this cycle.
        proposed_this_cycle: dict[str, Any] | None = None
        for tc in response.tool_calls:
            args = _parse_tool_args(tc["arguments"])
            try:
                result = execute_tool(tc["name"], args, tool_ctx)
                obs = Observation(tool_name=tc["name"], cycle=cycle, result=result)
            except ToolError as e:
                obs = Observation(tool_name=tc["name"], cycle=cycle, error=str(e))
            except Exception as e:  # never let a tool crash the loop
                log.exception("tool %s raised", tc["name"])
                obs = Observation(
                    tool_name=tc["name"],
                    cycle=cycle,
                    error=f"{type(e).__name__}: {str(e)[:200]}",
                )

            history.append(ToolCall(id=tc["id"], name=tc["name"], arguments=args), obs)

            if tc["name"] == "propose_fix" and not obs.is_error:
                proposed_this_cycle = args

        if proposed_this_cycle is not None:
            state.proposed_fixes.append(proposed_this_cycle)
            state.confidence = proposed_this_cycle.get("confidence", "medium")
            return state

    # Cycle budget exhausted without a fix proposal.
    state.confidence = "low"
    return state
