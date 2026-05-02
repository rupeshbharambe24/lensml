"""Unit tests for lensml.llm.client."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from lensml.llm.client import (
    AllRoutesFailedError,
    Attempt,
    LensmlLLM,
    LLMResponse,
)
from lensml.llm.routing import ProviderRoute, Task

# ---- helpers ----


def _fake_completion(content: str | None = "ok", tool_calls: list | None = None):
    """Build a fake LiteLLM completion object that quacks like the real one."""

    class _FakeFn:
        def __init__(self, name: str, arguments: str) -> None:
            self.name = name
            self.arguments = arguments

    class _FakeToolCall:
        def __init__(self, id_: str, name: str, arguments: str) -> None:
            self.id = id_
            self.function = _FakeFn(name, arguments)

    class _FakeMessage:
        def __init__(self) -> None:
            self.content = content
            self.tool_calls = (
                [_FakeToolCall(tc["id"], tc["name"], tc["arguments"]) for tc in tool_calls]
                if tool_calls
                else None
            )

    class _FakeChoice:
        def __init__(self) -> None:
            self.message = _FakeMessage()

    class _FakeResponse:
        def __init__(self) -> None:
            self.choices = [_FakeChoice()]

    return _FakeResponse()


def _llm(routes: dict | None = None, *, load_config: bool = False) -> LensmlLLM:
    """Test-friendly client: skip config loading by default (no ~/.lensml/config.toml)."""
    return LensmlLLM(
        config_path=Path("/tmp/nonexistent-lensml-config.toml"),
        routes=routes,
        load_config=load_config,
    )


_TEST_ROUTES = {
    Task.DIAGNOSIS: [
        ProviderRoute("primary", "fake/primary"),
        ProviderRoute("secondary", "fake/secondary"),
        ProviderRoute("tertiary", "fake/tertiary"),
    ],
}


# ---- complete() success path ----


@pytest.mark.asyncio
async def test_complete_uses_first_route_on_success() -> None:
    fake = _fake_completion(content="hello")
    with patch("lensml.llm.client.litellm.acompletion", new=AsyncMock(return_value=fake)) as mock:
        llm = _llm(routes=_TEST_ROUTES)
        resp = await llm.complete(Task.DIAGNOSIS, messages=[{"role": "user", "content": "hi"}])

        mock.assert_awaited_once()
        kwargs = mock.await_args.kwargs
        assert kwargs["model"] == "fake/primary"
        assert resp.label == "primary"
        assert resp.model == "fake/primary"
        assert resp.content == "hello"
        assert len(resp.attempts) == 1
        assert resp.attempts[0].ok is True


@pytest.mark.asyncio
async def test_complete_returns_normalized_tool_calls() -> None:
    fake = _fake_completion(
        content=None,
        tool_calls=[{"id": "tc-1", "name": "inspect_loss", "arguments": '{"window": 50}'}],
    )
    with patch("lensml.llm.client.litellm.acompletion", new=AsyncMock(return_value=fake)):
        llm = _llm(routes=_TEST_ROUTES)
        resp = await llm.complete(Task.DIAGNOSIS, messages=[{"role": "user", "content": "x"}])

        assert resp.content is None
        assert resp.tool_calls is not None
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0]["id"] == "tc-1"
        assert resp.tool_calls[0]["name"] == "inspect_loss"
        assert resp.tool_calls[0]["arguments"] == '{"window": 50}'


# ---- failover ----


@pytest.mark.asyncio
async def test_complete_falls_back_on_first_route_error() -> None:
    fake_ok = _fake_completion(content="from secondary")
    mock = AsyncMock(side_effect=[RuntimeError("rate limited"), fake_ok])
    with patch("lensml.llm.client.litellm.acompletion", new=mock):
        llm = _llm(routes=_TEST_ROUTES)
        resp = await llm.complete(Task.DIAGNOSIS, messages=[{"role": "user", "content": "x"}])

        assert mock.await_count == 2
        assert resp.label == "secondary"
        assert resp.content == "from secondary"
        assert len(resp.attempts) == 2
        assert resp.attempts[0].ok is False
        assert resp.attempts[0].error is not None
        assert "rate limited" in resp.attempts[0].error
        assert resp.attempts[1].ok is True


@pytest.mark.asyncio
async def test_complete_walks_full_chain() -> None:
    fake_ok = _fake_completion(content="from tertiary")
    mock = AsyncMock(
        side_effect=[
            RuntimeError("fail-1"),
            RuntimeError("fail-2"),
            fake_ok,
        ]
    )
    with patch("lensml.llm.client.litellm.acompletion", new=mock):
        llm = _llm(routes=_TEST_ROUTES)
        resp = await llm.complete(Task.DIAGNOSIS, messages=[{"role": "user", "content": "x"}])

        assert mock.await_count == 3
        assert resp.label == "tertiary"
        assert len(resp.attempts) == 3


@pytest.mark.asyncio
async def test_complete_raises_when_all_routes_fail() -> None:
    mock = AsyncMock(side_effect=[RuntimeError("a"), RuntimeError("b"), RuntimeError("c")])
    with patch("lensml.llm.client.litellm.acompletion", new=mock):
        llm = _llm(routes=_TEST_ROUTES)
        with pytest.raises(AllRoutesFailedError) as exc_info:
            await llm.complete(Task.DIAGNOSIS, messages=[{"role": "user", "content": "x"}])
        assert exc_info.value.task == Task.DIAGNOSIS
        assert len(exc_info.value.attempts) == 3


@pytest.mark.asyncio
async def test_complete_raises_for_task_with_no_routes() -> None:
    llm = _llm(routes={})
    with pytest.raises(AllRoutesFailedError):
        await llm.complete(Task.DIAGNOSIS, messages=[{"role": "user", "content": "x"}])


# ---- request shape ----


@pytest.mark.asyncio
async def test_complete_passes_tools_and_tool_choice() -> None:
    fake = _fake_completion(content="ok")
    mock = AsyncMock(return_value=fake)
    with patch("lensml.llm.client.litellm.acompletion", new=mock):
        llm = _llm(routes=_TEST_ROUTES)
        tools = [{"type": "function", "function": {"name": "x", "parameters": {}}}]
        await llm.complete(
            Task.DIAGNOSIS,
            messages=[{"role": "user", "content": "x"}],
            tools=tools,
        )
        kwargs = mock.await_args.kwargs
        assert kwargs["tools"] == tools
        assert kwargs["tool_choice"] == "auto"


@pytest.mark.asyncio
async def test_complete_omits_tools_when_none() -> None:
    fake = _fake_completion(content="ok")
    mock = AsyncMock(return_value=fake)
    with patch("lensml.llm.client.litellm.acompletion", new=mock):
        llm = _llm(routes=_TEST_ROUTES)
        await llm.complete(Task.DIAGNOSIS, messages=[{"role": "user", "content": "x"}])
        kwargs = mock.await_args.kwargs
        assert "tools" not in kwargs


@pytest.mark.asyncio
async def test_openrouter_route_adds_app_headers() -> None:
    fake = _fake_completion(content="ok")
    mock = AsyncMock(return_value=fake)
    routes = {Task.DIAGNOSIS: [ProviderRoute("or", "openrouter/anything/x:free")]}
    with patch("lensml.llm.client.litellm.acompletion", new=mock):
        llm = _llm(routes=routes)
        await llm.complete(Task.DIAGNOSIS, messages=[{"role": "user", "content": "x"}])
        kwargs = mock.await_args.kwargs
        assert "extra_headers" in kwargs
        assert "HTTP-Referer" in kwargs["extra_headers"]
        assert "X-Title" in kwargs["extra_headers"]


@pytest.mark.asyncio
async def test_complete_forwards_extra_kwargs() -> None:
    fake = _fake_completion(content="ok")
    mock = AsyncMock(return_value=fake)
    with patch("lensml.llm.client.litellm.acompletion", new=mock):
        llm = _llm(routes=_TEST_ROUTES)
        await llm.complete(
            Task.DIAGNOSIS,
            messages=[{"role": "user", "content": "x"}],
            seed=42,
        )
        kwargs = mock.await_args.kwargs
        assert kwargs["seed"] == 42


# ---- config loading ----


def test_config_loading_skipped_when_file_missing() -> None:
    """Missing config file is OK — should not raise."""
    llm = LensmlLLM(
        config_path=Path("/tmp/nonexistent-lensml-config.toml"),
        load_config=True,
    )
    assert isinstance(llm, LensmlLLM)


def test_config_loading_does_not_overwrite_existing_env_vars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If GROQ_API_KEY is already set (CI secret), config.toml must not overwrite."""
    cfg = tmp_path / "config.toml"
    cfg.write_text('[providers.groq]\napi_key = "from_config"\n')

    monkeypatch.setenv("GROQ_API_KEY", "from_env_already_set")
    llm = LensmlLLM(config_path=cfg, load_config=True)

    import os

    assert os.environ["GROQ_API_KEY"] == "from_env_already_set"
    del llm  # silence unused-warning


def test_config_loading_sets_unset_env_vars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text('[providers.cerebras]\napi_key = "csk-from-config"\n')

    monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)
    LensmlLLM(config_path=cfg, load_config=True)

    import os

    assert os.environ.get("CEREBRAS_API_KEY") == "csk-from-config"


# ---- LLMResponse / Attempt dataclasses ----


def test_attempt_to_dict() -> None:
    a = Attempt(label="x", model="x/y", ok=True, latency_ms=42.123)
    d = a.to_dict()
    assert d["label"] == "x"
    assert d["model"] == "x/y"
    assert d["ok"] is True
    assert d["latency_ms"] == 42.12  # rounded
    assert d["error"] is None


def test_llm_response_default_attempts_empty() -> None:
    r = LLMResponse(content="x", tool_calls=None, model="m", label="l", latency_ms=1.0)
    assert r.attempts == []
