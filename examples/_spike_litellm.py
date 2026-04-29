"""Day 0.5 LiteLLM tool-call portability spike.

Goal (from plan §11 risk #5):
    Prove cross-provider tool-call schema portability via LiteLLM. One tool
    definition, multiple providers — confirms LiteLLM normalizes the
    OpenAI-compat (Groq, Cerebras, Mistral, OpenRouter) and Google-native
    (Gemini) tool-calling shapes into a single API.

Plan refs:
    §5.5 — per-task primary/secondary/tertiary routing across providers.
           A single tool registry must work against all of them via LiteLLM.
    §11 risk #5 — cross-provider tool-call schema portability is the
           bet that makes plan §5.5 viable. If LiteLLM doesn't normalize
           cleanly, fallback is to pin one provider (Groq Llama 3.3 70B).

Reads keys from: ~/.lensml/config.toml (created in step 0.3).

Run:
    python examples/_spike_litellm.py

Exit codes:
    0  >= 4 of 5 providers returned valid tool_calls
    1  < 4 providers worked
"""

from __future__ import annotations

import os
import sys
import time
import tomllib
from pathlib import Path

import litellm

# Silence litellm's own logging
litellm.suppress_debug_info = True
os.environ.setdefault("LITELLM_LOG", "ERROR")

CONFIG_PATH = Path.home() / ".lensml" / "config.toml"

# Models verified working in step 0.3 — see ~/.lensml/notes/limits-2026-04-29.md
PROVIDERS = [
    # (litellm_model_string, env_var_for_key, friendly_name)
    ("groq/llama-3.1-8b-instant", "GROQ_API_KEY", "Groq"),
    ("cerebras/llama3.1-8b", "CEREBRAS_API_KEY", "Cerebras"),
    ("mistral/mistral-small-latest", "MISTRAL_API_KEY", "Mistral"),
    ("gemini/gemini-2.5-flash", "GEMINI_API_KEY", "Gemini"),
    # OpenRouter: nemotron-nano-9b worked in step 0.3 with full tool_calls
    ("openrouter/nvidia/nemotron-nano-9b-v2:free", "OPENROUTER_API_KEY", "OpenRouter"),
]

# Single tool definition — LiteLLM accepts the OpenAI-compat shape and
# translates to Gemini's `function_declarations` automatically.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current time in a given city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The city name"},
                },
                "required": ["city"],
            },
        },
    },
]

MESSAGES = [
    {"role": "user", "content": "What time is it in Tokyo? Call the get_current_time tool."},
]


def load_keys() -> None:
    """Read keys from ~/.lensml/config.toml and export as env vars (LiteLLM convention)."""
    if not CONFIG_PATH.exists():
        print(f"FAIL: config not found at {CONFIG_PATH}")
        sys.exit(2)
    with CONFIG_PATH.open("rb") as f:
        cfg = tomllib.load(f)
    providers = cfg.get("providers", {})
    mapping = {
        "groq": "GROQ_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "cerebras": "CEREBRAS_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "mistral": "MISTRAL_API_KEY",
    }
    for cfg_name, env_var in mapping.items():
        key = providers.get(cfg_name, {}).get("api_key", "")
        if key:
            os.environ[env_var] = key


def test_provider(model: str, label: str) -> tuple[bool, str]:
    """Call litellm.completion with tools. Return (ok, summary)."""
    t0 = time.perf_counter()
    try:
        # OpenRouter requires app identification headers
        extra = {}
        if model.startswith("openrouter/"):
            extra["extra_headers"] = {
                "HTTP-Referer": "https://github.com/rupeshbharambe24/lensml",
                "X-Title": "lensml",
            }
        resp = litellm.completion(
            model=model,
            messages=MESSAGES,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=200,
            timeout=45,
            **extra,
        )
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return False, f"FAIL after {elapsed:.2f}s: {type(e).__name__}: {str(e)[:200]}"

    elapsed = time.perf_counter() - t0
    msg = resp.choices[0].message
    tool_calls = getattr(msg, "tool_calls", None)
    # Some providers may put it elsewhere; try the dict path
    if not tool_calls and hasattr(msg, "model_dump"):
        d = msg.model_dump()
        tool_calls = d.get("tool_calls")
    if not tool_calls:
        return False, f"FAIL after {elapsed:.2f}s: no tool_calls in response"

    tc = tool_calls[0]
    name = tc.function.name if hasattr(tc, "function") else tc["function"]["name"]
    args = tc.function.arguments if hasattr(tc, "function") else tc["function"]["arguments"]
    return True, f"OK in {elapsed:.2f}s: {name}({args})"


def main() -> int:
    print("=== lensml LiteLLM tool-call portability spike ===")
    print(f"litellm: {__import__('importlib.metadata', fromlist=['version']).version('litellm')}")
    print(f"config:  {CONFIG_PATH}")
    print()
    load_keys()
    print(
        f"loaded {sum(1 for _, k, _ in PROVIDERS if os.environ.get(k))}/{len(PROVIDERS)} keys from config"
    )
    print()

    print("tool definition (OpenAI-compat schema, LiteLLM normalizes for Gemini):")
    print("    get_current_time(city: string)")
    print()
    print("prompt: 'What time is it in Tokyo? Call the get_current_time tool.'")
    print()

    results: list[tuple[str, str, bool, str]] = []
    for model, key_env, label in PROVIDERS:
        if not os.environ.get(key_env):
            results.append((label, model, False, f"SKIP: {key_env} not set"))
            continue
        print(f"--- {label}  ({model}) ---")
        ok, summary = test_provider(model, label)
        print(f"  {summary}")
        print()
        results.append((label, model, ok, summary))

    print("=== summary ===")
    n_ok = sum(1 for _, _, ok, _ in results if ok)
    n_total = len(results)
    for label, model, ok, _summary in results:
        mark = "OK  " if ok else "FAIL"
        print(f"  [{mark}] {label:12s} {model}")
    print()
    print(f"verdict: {n_ok}/{n_total} providers returned a structured tool_call via LiteLLM")
    print()

    if n_ok >= 4:
        print(f"PASS: {n_ok}/{n_total} providers — cross-provider tool-call portability confirmed")
        return 0
    print(
        f"FAIL: only {n_ok}/{n_total} providers worked; pin to single provider per plan §11 risk #5"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
