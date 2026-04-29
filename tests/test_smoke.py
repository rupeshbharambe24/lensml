"""Smoke test — package imports cleanly and exposes the v1 public surface.

Per plan §1: the user-facing API is `lensml.watch(model, optimizer, dataloader)`.
This test confirms the import path is clean. Real behavior tests land in
later phases as the implementations are filled in.
"""

from __future__ import annotations


def test_package_imports() -> None:
    import lensml  # noqa: F401


def test_version_is_set() -> None:
    import lensml

    assert isinstance(lensml.__version__, str)
    assert len(lensml.__version__) > 0


def test_public_api_surface() -> None:
    import lensml

    assert hasattr(lensml, "watch")
    assert hasattr(lensml, "unwatch")
    assert callable(lensml.watch)
    assert callable(lensml.unwatch)
