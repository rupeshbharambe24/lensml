"""Per-task LLM provider routing — plan §5.5.

Each task type has a primary / secondary / tertiary route. On error
(rate-limit, timeout, upstream 5xx) the client walks down the chain
until something succeeds.

The defaults below were verified working in step 0.5 of the
implementation sequence (see `~/.lensml/notes/limits-2026-04-29.md`).
"""

from __future__ import annotations

from enum import StrEnum
from typing import NamedTuple


class Task(StrEnum):
    """Categories of LLM call. Each has its own fallback chain."""

    ANOMALY_CLASSIFICATION = "anomaly_classification"
    DIAGNOSIS = "diagnosis"
    PATCH_GENERATION = "patch_generation"
    VERIFICATION = "verification"


class ProviderRoute(NamedTuple):
    """One entry in a fallback chain.

    `label`: short slug used in logs / event-store routing decisions.
    `model`: LiteLLM model string (provider/model[:variant]).
    """

    label: str
    model: str


# Plan §5.5 routing table. Pairs of (label, litellm-model-string).
DEFAULT_ROUTES: dict[Task, list[ProviderRoute]] = {
    # Frequent, light prompt — Groq's regional latency shines here.
    Task.ANOMALY_CLASSIFICATION: [
        ProviderRoute("groq-llama-8b", "groq/llama-3.1-8b-instant"),
        ProviderRoute("cerebras-llama-8b", "cerebras/llama3.1-8b"),
        ProviderRoute("gemini-flash-lite", "gemini/gemini-2.5-flash-lite"),
    ],
    # Heavyweight reasoning — prefer thinking-trace models.
    Task.DIAGNOSIS: [
        ProviderRoute("gemini-pro", "gemini/gemini-2.5-pro"),
        ProviderRoute("cerebras-llama-70b", "cerebras/llama-3.3-70b"),
        ProviderRoute(
            "openrouter-nemotron-super",
            "openrouter/nvidia/nemotron-3-super-120b-a12b:free",
        ),
    ],
    # Code-specialized models for libcst patch generation.
    Task.PATCH_GENERATION: [
        ProviderRoute("openrouter-qwen-coder", "openrouter/qwen/qwen3-coder:free"),
        ProviderRoute("mistral-codestral", "mistral/codestral-latest"),
        ProviderRoute("gemini-pro", "gemini/gemini-2.5-pro"),
    ],
    # "Does this fix make sense?" — needs reasoning, not code.
    Task.VERIFICATION: [
        ProviderRoute("cerebras-llama-70b", "cerebras/llama-3.3-70b"),
        ProviderRoute("gemini-pro", "gemini/gemini-2.5-pro"),
        ProviderRoute("openrouter-gpt-oss-120b", "openrouter/openai/gpt-oss-120b:free"),
    ],
}


def get_routes(
    task: Task,
    routes: dict[Task, list[ProviderRoute]] | None = None,
) -> list[ProviderRoute]:
    """Return the fallback chain for `task` (default `DEFAULT_ROUTES`)."""
    table = routes if routes is not None else DEFAULT_ROUTES
    return table.get(task, [])
