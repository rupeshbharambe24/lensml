"""Unit tests for DataLoaderWorkers (plan §4 row 5)."""

from __future__ import annotations

import libcst as cst

from lensml.fixes.dataloader_workers import DataLoaderWorkers


def _round_trip(src: str) -> None:
    cst.parse_module(src)


def test_appends_kwargs_when_none_present() -> None:
    src = "loader = DataLoader(dataset, batch_size=32)\n"
    out = DataLoaderWorkers(num_workers=4).apply(src)
    assert out.applied is True
    assert "num_workers=4" in out.new_source
    assert "pin_memory=True" in out.new_source
    assert "persistent_workers=True" in out.new_source
    _round_trip(out.new_source)


def test_replaces_existing_num_workers() -> None:
    src = "loader = DataLoader(dataset, batch_size=32, num_workers=0)\n"
    out = DataLoaderWorkers(num_workers=4).apply(src)
    assert out.applied is True
    assert "num_workers=4" in out.new_source
    assert "num_workers=0" not in out.new_source


def test_idempotent_when_already_configured() -> None:
    src = (
        "loader = DataLoader(dataset, batch_size=32, "
        "num_workers=4, pin_memory=True, persistent_workers=True)\n"
    )
    out = DataLoaderWorkers(num_workers=4).apply(src)
    assert out.applied is False


def test_handles_attribute_access() -> None:
    """`torch.utils.data.DataLoader(...)` is recognized too."""
    src = "loader = torch.utils.data.DataLoader(dataset, batch_size=32)\n"
    out = DataLoaderWorkers().apply(src)
    assert out.applied is True
    assert "num_workers=" in out.new_source


def test_skips_unrelated_call() -> None:
    src = "result = foo(dataset, batch_size=32)\n"
    out = DataLoaderWorkers().apply(src)
    assert out.applied is False


def test_multiple_dataloader_calls_all_updated() -> None:
    src = (
        "train_loader = DataLoader(train_ds, batch_size=32)\n"
        "val_loader = DataLoader(val_ds, batch_size=64)\n"
    )
    out = DataLoaderWorkers().apply(src)
    assert out.applied is True
    assert out.new_source.count("num_workers=") == 2
    assert out.new_source.count("pin_memory=True") == 2


def test_parse_error_returns_unapplied() -> None:
    out = DataLoaderWorkers().apply("def broken(")
    assert out.applied is False


def test_diff_when_applied() -> None:
    src = "loader = DataLoader(dataset)\n"
    out = DataLoaderWorkers().apply(src)
    assert out.applied is True
    assert "num_workers=" in out.diff
