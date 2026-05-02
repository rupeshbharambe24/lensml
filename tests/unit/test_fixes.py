"""Unit tests for libcst fix templates (lensml.fixes.*)."""

from __future__ import annotations

import libcst as cst

from lensml.fixes.add_normalize import AddNormalize
from lensml.fixes.base import FixResult
from lensml.fixes.grad_clip import AddGradientClipping
from lensml.fixes.lower_lr import LowerLearningRate
from lensml.fixes.replace_activation import ReplaceActivation


def _ast_eq(a: str, b: str) -> bool:
    """Compare two source strings via libcst — ignores incidental whitespace."""
    return cst.parse_module(a).deep_equals(cst.parse_module(b))


# ============================================================================
# FixResult / base behavior
# ============================================================================


def test_fix_result_diff_empty_when_not_applied() -> None:
    r = FixResult(applied=False, template="X", old_source="a = 1\n")
    assert r.diff == ""


def test_fix_result_diff_renders_when_applied() -> None:
    r = FixResult(
        applied=True,
        template="X",
        old_source="a = 1\n",
        new_source="a = 2\n",
    )
    assert "-a = 1" in r.diff
    assert "+a = 2" in r.diff


# ============================================================================
# AddGradientClipping
# ============================================================================


def test_grad_clip_inserts_between_backward_and_step() -> None:
    src = (
        "def train_step(model, optimizer, loss):\n"
        "    optimizer.zero_grad()\n"
        "    loss.backward()\n"
        "    optimizer.step()\n"
    )
    fix = AddGradientClipping(max_norm=1.0, model_name="model")
    r = fix.apply(src)
    assert r.applied
    assert "torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)" in r.new_source

    # The clip line must appear between backward and step
    lines = r.new_source.splitlines()
    backward_idx = next(i for i, line in enumerate(lines) if "loss.backward()" in line)
    clip_idx = next(i for i, line in enumerate(lines) if "clip_grad_norm_" in line)
    step_idx = next(i for i, line in enumerate(lines) if "optimizer.step()" in line)
    assert backward_idx < clip_idx < step_idx


def test_grad_clip_no_pattern_returns_not_applied() -> None:
    src = "x = 1\n"
    r = AddGradientClipping().apply(src)
    assert not r.applied
    assert "no `.backward()`" in (r.reason or "")


def test_grad_clip_handles_parse_error_gracefully() -> None:
    r = AddGradientClipping().apply("def broken(:\n")
    assert not r.applied
    assert "parse error" in (r.reason or "")


def test_grad_clip_respects_custom_max_norm() -> None:
    src = "def step(loss, opt):\n    loss.backward()\n    opt.step()\n"
    r = AddGradientClipping(max_norm=2.5, model_name="net").apply(src)
    assert r.applied
    assert "max_norm=2.5" in r.new_source
    assert "net.parameters()" in r.new_source


def test_grad_clip_skips_when_not_immediately_followed_by_step() -> None:
    """If backward is the last statement (no step after), don't insert."""
    src = "def f(loss):\n    loss.backward()\n"
    r = AddGradientClipping().apply(src)
    assert not r.applied


# ============================================================================
# LowerLearningRate
# ============================================================================


def test_lower_lr_scales_float_literal() -> None:
    src = "opt = torch.optim.SGD(model.parameters(), lr=0.01)\n"
    r = LowerLearningRate(factor=0.3).apply(src)
    assert r.applied
    # 0.01 * 0.3 = 0.003
    assert "lr=0.003" in r.new_source


def test_lower_lr_scales_integer_literal() -> None:
    src = "opt = torch.optim.SGD(p, lr=1)\n"
    r = LowerLearningRate(factor=0.5).apply(src)
    assert r.applied
    assert "lr=0.5" in r.new_source


def test_lower_lr_wraps_non_literal_expression() -> None:
    src = "opt = SGD(p, lr=base_lr)\n"
    r = LowerLearningRate(factor=0.3).apply(src)
    assert r.applied
    # Non-literal → wrapped: lr=base_lr * 0.3
    assert "base_lr" in r.new_source
    assert "0.3" in r.new_source


def test_lower_lr_returns_unmodified_when_no_lr_kwarg() -> None:
    src = "opt = SGD(p)\n"
    r = LowerLearningRate().apply(src)
    assert not r.applied


def test_lower_lr_factor_validation() -> None:
    import pytest

    with pytest.raises(ValueError):
        LowerLearningRate(factor=0)
    with pytest.raises(ValueError):
        LowerLearningRate(factor=-1)


# ============================================================================
# ReplaceActivation
# ============================================================================


def test_replace_activation_relu_to_leaky() -> None:
    src = (
        "import torch.nn as nn\n"
        "class M(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.act = nn.ReLU()\n"
    )
    r = ReplaceActivation(old="ReLU", new="LeakyReLU", new_args="0.01").apply(src)
    assert r.applied
    assert "nn.LeakyReLU(0.01)" in r.new_source
    assert "nn.ReLU()" not in r.new_source


def test_replace_activation_sigmoid_to_gelu() -> None:
    src = "act = nn.Sigmoid()\n"
    r = ReplaceActivation(old="Sigmoid", new="GELU", new_args="").apply(src)
    assert r.applied
    assert "nn.GELU()" in r.new_source
    assert "nn.Sigmoid" not in r.new_source


def test_replace_activation_bare_name() -> None:
    """Handles `from torch.nn import ReLU; ReLU()` style too."""
    src = "from torch.nn import ReLU\nact = ReLU()\n"
    r = ReplaceActivation(old="ReLU", new="LeakyReLU", new_args="0.01").apply(src)
    assert r.applied
    assert "LeakyReLU(0.01)" in r.new_source


def test_replace_activation_no_match() -> None:
    src = "act = nn.GELU()\n"
    r = ReplaceActivation(old="ReLU", new="LeakyReLU").apply(src)
    assert not r.applied


def test_replace_activation_preserves_other_calls() -> None:
    """Don't touch unrelated calls like nn.Linear()."""
    src = "linear = nn.Linear(10, 20); act = nn.ReLU()\n"
    r = ReplaceActivation(old="ReLU", new="LeakyReLU", new_args="0.01").apply(src)
    assert r.applied
    assert "nn.Linear(10, 20)" in r.new_source
    assert "nn.LeakyReLU(0.01)" in r.new_source


# ============================================================================
# AddNormalize
# ============================================================================


def test_add_normalize_appends_to_compose() -> None:
    src = (
        "import torchvision.transforms as transforms\n"
        "tfm = transforms.Compose([\n"
        "    transforms.Resize(32),\n"
        "    transforms.ToTensor(),\n"
        "])\n"
    )
    r = AddNormalize().apply(src)
    assert r.applied
    assert "transforms.Normalize" in r.new_source
    assert "0.485" in r.new_source
    assert "0.229" in r.new_source
    # Existing transforms preserved
    assert "transforms.Resize(32)" in r.new_source
    assert "transforms.ToTensor()" in r.new_source


def test_add_normalize_idempotent_when_already_present() -> None:
    src = (
        "tfm = transforms.Compose([\n"
        "    transforms.ToTensor(),\n"
        "    transforms.Normalize(mean=[0.5], std=[0.5]),\n"
        "])\n"
    )
    r = AddNormalize().apply(src)
    assert not r.applied
    assert "Normalize already present" in (r.reason or "") or (
        "no `transforms.Compose" in (r.reason or "")
    )


def test_add_normalize_no_compose_call() -> None:
    src = "x = 1\n"
    r = AddNormalize().apply(src)
    assert not r.applied


def test_add_normalize_custom_stats() -> None:
    src = "tfm = transforms.Compose([transforms.ToTensor()])\n"
    r = AddNormalize(mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25)).apply(src)
    assert r.applied
    assert "0.5" in r.new_source
    assert "0.25" in r.new_source


# ============================================================================
# Round-trip: parse → apply → parse must succeed for every fix
# ============================================================================


def test_grad_clip_output_parses_back() -> None:
    src = "def f(loss, opt):\n    loss.backward()\n    opt.step()\n"
    r = AddGradientClipping().apply(src)
    assert r.applied
    cst.parse_module(r.new_source)  # must not raise


def test_lower_lr_output_parses_back() -> None:
    src = "opt = SGD(p, lr=0.01)\n"
    r = LowerLearningRate().apply(src)
    assert r.applied
    cst.parse_module(r.new_source)


def test_replace_activation_output_parses_back() -> None:
    src = "act = nn.ReLU()\n"
    r = ReplaceActivation().apply(src)
    assert r.applied
    cst.parse_module(r.new_source)


def test_add_normalize_output_parses_back() -> None:
    src = "tfm = transforms.Compose([transforms.ToTensor()])\n"
    r = AddNormalize().apply(src)
    assert r.applied
    cst.parse_module(r.new_source)
