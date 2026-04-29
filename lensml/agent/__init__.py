"""lensml.agent — single-agent ReAct loop, ~80 lines, hand-rolled.

Per plan §5.1: no LangChain, no LangGraph, no Pydantic AI. Reasons:
    (1) every framework adds debug surface area,
    (2) solo + 30 days = control matters more than abstraction,
    (3) the loop is genuinely small.

Module map (plan §3.2):
    loop.py       ReAct loop
    tools.py      tool registry (12 tools, all read-only except propose_fix)
    memory.py     three-layer memory: working / sliding-mask / persistent state
    confidence.py self-consistency at N=3, temperature 0.7
    prompts/      system prompts per task
"""
