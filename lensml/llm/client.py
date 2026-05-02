"""LiteLLM-backed multi-provider client with per-task fallback.

Plan refs:
    §5.5 — per-task primary/secondary/tertiary routing across 5 free-tier
           providers. On any 4xx/5xx/timeout, retry the next route.
    §11 risk #5 — cross-provider tool-call schema portability mitigated
           via LiteLLM (validated in step 0.5 spike).

Reads provider keys from ``~/.lensml/config.toml`` (or ``$LENSML_CONFIG``)
and exposes them as env vars LiteLLM understands. Existing env vars are
NOT overwritten — useful for CI where keys come from secrets.

Async-only: the agent loop and FastAPI server are both async, so calling
``litellm.acompletion`` directly avoids running tasks in executor threads.
"""

from __future__ import annotations

import logging
import os
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import litellm

from lensml.llm.routing import DEFAULT_ROUTES, ProviderRoute, Task, get_routes

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path.home() / ".lensml" / "config.toml"

# Provider section name in config.toml → LiteLLM env var name.
PROVIDER_ENV_VARS: dict[str, str] = {
    "groq": "GROQ_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "mistral": "MISTRAL_API_KEY",
}


@dataclass
class Attempt:
    """One entry in the per-call attempt log (for event-store routing
    decisions, plan §5.5)."""

    label: str
    model: str
    ok: bool
    latency_ms: float
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "model": self.model,
            "ok": self.ok,
            "latency_ms": round(self.latency_ms, 2),
            "error": self.error,
        }


@dataclass
class LLMResponse:
    """Normalized response wrapper around a raw LiteLLM completion.

    `content` is None when the model returned tool calls only.
    `tool_calls` is None when the model returned plain text only.
    `attempts` lists every route tried (last entry is the success).
    """

    content: str | None
    tool_calls: list[dict[str, Any]] | None
    model: str
    label: str
    latency_ms: float
    attempts: list[Attempt] = field(default_factory=list)
    raw: Any = None


class AllRoutesFailedError(RuntimeError):
    """Raised when every route in a task's fallback chain failed."""

    def __init__(self, task: Task, attempts: list[Attempt]) -> None:
        last_err = attempts[-1].error if attempts else "no routes configured"
        super().__init__(
            f"All {len(attempts)} provider routes failed for task "
            f"'{task.value}'; last error: {last_err}"
        )
        self.task = task
        self.attempts = attempts


class LensmlLLM:
    """Multi-provider LLM client with per-task fallback.

    Construct once, call ``complete()`` for every agent step. The
    instance is safe to share across coroutines.
    """

    def __init__(
        self,
        *,
        config_path: Path | None = None,
        routes: dict[Task, list[ProviderRoute]] | None = None,
        timeout: float = 30.0,
        load_config: bool = True,
    ) -> None:
        self.config_path = config_path or self._default_config_path()
        self.routes = routes if routes is not None else DEFAULT_ROUTES
        self.timeout = timeout
        if load_config:
            self._load_config()

    # ---- config loading ----

    @staticmethod
    def _default_config_path() -> Path:
        env = os.environ.get("LENSML_CONFIG")
        return Path(env) if env else DEFAULT_CONFIG_PATH

    def _load_config(self) -> None:
        """Read config.toml and export keys as env vars (LiteLLM convention).

        Existing env vars are NOT overwritten — CI/secrets take precedence.
        """
        if not self.config_path.exists():
            log.info("LensmlLLM: config not found at %s (skipping)", self.config_path)
            return
        try:
            with self.config_path.open("rb") as f:
                cfg = tomllib.load(f)
        except Exception:
            log.exception("failed to load %s", self.config_path)
            return
        providers = cfg.get("providers", {})
        for label, env_var in PROVIDER_ENV_VARS.items():
            section = providers.get(label, {})
            key = section.get("api_key", "") if isinstance(section, dict) else ""
            if key and not os.environ.get(env_var):
                os.environ[env_var] = key

    # ---- public API ----

    async def complete(
        self,
        task: Task,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        **extra: Any,
    ) -> LLMResponse:
        """Try each route in the task's fallback chain until one succeeds.

        Raises :class:`AllRoutesFailedError` if every route fails.
        """
        routes = get_routes(task, self.routes)
        if not routes:
            raise AllRoutesFailedError(task, [])

        attempts: list[Attempt] = []
        for route in routes:
            t0 = time.perf_counter()
            try:
                raw = await self._call_litellm(
                    route,
                    messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **extra,
                )
            except Exception as e:  # broad catch is intentional — any error → next route
                latency_ms = (time.perf_counter() - t0) * 1000
                attempts.append(
                    Attempt(
                        label=route.label,
                        model=route.model,
                        ok=False,
                        latency_ms=latency_ms,
                        error=f"{type(e).__name__}: {str(e)[:200]}",
                    )
                )
                log.warning("LLM call failed via %s: %s: %s", route.label, type(e).__name__, e)
                continue

            latency_ms = (time.perf_counter() - t0) * 1000
            attempts.append(
                Attempt(
                    label=route.label,
                    model=route.model,
                    ok=True,
                    latency_ms=latency_ms,
                )
            )
            return self._build_response(raw, route, latency_ms, attempts)

        raise AllRoutesFailedError(task, attempts)

    # ---- internal ----

    async def _call_litellm(
        self,
        route: ProviderRoute,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        **extra: Any,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": route.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "timeout": self.timeout,
        }
        if tools is not None:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice if tool_choice is not None else "auto"

        # OpenRouter requires app identification headers (per their TOS).
        if route.model.startswith("openrouter/"):
            kwargs["extra_headers"] = {
                "HTTP-Referer": "https://github.com/rupeshbharambe24/lensml",
                "X-Title": "lensml",
            }

        kwargs.update(extra)
        return await litellm.acompletion(**kwargs)

    @staticmethod
    def _build_response(
        raw: Any,
        route: ProviderRoute,
        latency_ms: float,
        attempts: list[Attempt],
    ) -> LLMResponse:
        msg = raw.choices[0].message
        content = getattr(msg, "content", None)
        tool_calls_raw = getattr(msg, "tool_calls", None)

        tool_calls: list[dict[str, Any]] | None = None
        if tool_calls_raw:
            tool_calls = []
            for tc in tool_calls_raw:
                fn = getattr(tc, "function", None)
                if fn is None:
                    continue
                tool_calls.append(
                    {
                        "id": getattr(tc, "id", "") or "",
                        "name": getattr(fn, "name", "") or "",
                        "arguments": getattr(fn, "arguments", "{}") or "{}",
                    }
                )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            model=route.model,
            label=route.label,
            latency_ms=latency_ms,
            attempts=attempts,
            raw=raw,
        )
