"""Unit tests for lensml.llm.routing."""

from __future__ import annotations

from lensml.llm.routing import DEFAULT_ROUTES, ProviderRoute, Task, get_routes


def test_all_tasks_have_routes() -> None:
    for task in Task:
        assert task in DEFAULT_ROUTES
        assert len(DEFAULT_ROUTES[task]) >= 1


def test_routes_have_three_tiers_for_main_tasks() -> None:
    """Plan §5.5 calls for primary/secondary/tertiary on the main four tasks."""
    for task in (Task.ANOMALY_CLASSIFICATION, Task.DIAGNOSIS, Task.PATCH_GENERATION):
        assert len(DEFAULT_ROUTES[task]) >= 3


def test_get_routes_returns_default() -> None:
    routes = get_routes(Task.DIAGNOSIS)
    assert routes[0].label == "gemini-pro"
    assert routes[0].model == "gemini/gemini-2.5-pro"


def test_get_routes_with_custom_table() -> None:
    custom = {Task.DIAGNOSIS: [ProviderRoute("test", "test/model")]}
    routes = get_routes(Task.DIAGNOSIS, custom)
    assert len(routes) == 1
    assert routes[0].label == "test"


def test_get_routes_unknown_task_returns_empty() -> None:
    """If a custom table doesn't list the task, we get []."""
    routes = get_routes(Task.DIAGNOSIS, {})
    assert routes == []


def test_provider_route_namedtuple() -> None:
    """ProviderRoute is a NamedTuple — supports .label / .model + indexing."""
    r = ProviderRoute("x", "y/z")
    assert r.label == "x"
    assert r.model == "y/z"
    assert r[0] == "x"
    assert r[1] == "y/z"
