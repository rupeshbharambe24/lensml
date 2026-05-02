"""Unit tests for the fix registry (lensml.fixes.__init__)."""

from __future__ import annotations

import pytest

from lensml.fixes import (
    FIX_REGISTRY,
    AddGradientClipping,
    AddNormalize,
    LowerLearningRate,
    ReplaceActivation,
    build_fix,
)


def test_registry_contains_all_v1_templates() -> None:
    assert "AddGradientClipping" in FIX_REGISTRY
    assert "LowerLearningRate" in FIX_REGISTRY
    assert "ReplaceActivation" in FIX_REGISTRY
    assert "AddNormalize" in FIX_REGISTRY
    assert FIX_REGISTRY["AddGradientClipping"] is AddGradientClipping
    assert FIX_REGISTRY["LowerLearningRate"] is LowerLearningRate
    assert FIX_REGISTRY["ReplaceActivation"] is ReplaceActivation
    assert FIX_REGISTRY["AddNormalize"] is AddNormalize


def test_build_fix_constructs_template() -> None:
    fix = build_fix("AddGradientClipping", {"max_norm": 2.0, "model_name": "net"})
    assert isinstance(fix, AddGradientClipping)
    assert fix.max_norm == 2.0
    assert fix.model_name == "net"


def test_build_fix_default_params() -> None:
    fix = build_fix("AddGradientClipping")
    assert isinstance(fix, AddGradientClipping)


def test_build_fix_unknown_template_raises() -> None:
    with pytest.raises(ValueError, match="unknown fix template"):
        build_fix("DoesNotExist")


def test_build_fix_invalid_params_raises() -> None:
    # AddGradientClipping doesn't accept `garbage` kwarg
    with pytest.raises(ValueError, match="invalid params"):
        build_fix("AddGradientClipping", {"garbage": True})
