"""Unit tests for lensml.agent.confidence."""

from __future__ import annotations

from lensml.agent.confidence import confidence_allows_apply, consensus


def test_consensus_unanimous_n3_is_high() -> None:
    label, conf = consensus(["exploding_grad", "exploding_grad", "exploding_grad"])
    assert label == "exploding_grad"
    assert conf == "high"


def test_consensus_majority_n3_is_medium() -> None:
    label, conf = consensus(["exploding_grad", "exploding_grad", "dead_relu"])
    assert label == "exploding_grad"
    assert conf == "medium"


def test_consensus_split_n3_is_low() -> None:
    label, conf = consensus(["a", "b", "c"])
    assert label is None
    assert conf == "low"


def test_consensus_empty_is_low() -> None:
    label, conf = consensus([])
    assert label is None
    assert conf == "low"


def test_consensus_single_n1_does_not_promote_to_high() -> None:
    """Plan §5.4 needs N≥2 for HIGH (a sample of 1 is not 'unanimous')."""
    label, conf = consensus(["x"])
    assert label == "x"
    # With n=1, count==n_total but n_total<2 → doesn't qualify as high
    assert conf == "medium"


def test_consensus_drops_none_labels() -> None:
    label, conf = consensus(["x", None, "x"])
    assert label == "x"
    # Two valid samples, both agree → unanimous → high (n_total>=2)
    assert conf == "high"


def test_consensus_drops_empty_strings() -> None:
    label, conf = consensus(["", "x", "x"])
    assert label == "x"
    assert conf == "high"  # only 2 valid, both agree


def test_consensus_unanimous_n2_is_high() -> None:
    label, conf = consensus(["x", "x"])
    assert label == "x"
    assert conf == "high"


def test_confidence_allows_apply() -> None:
    assert confidence_allows_apply("high") is True
    assert confidence_allows_apply("medium") is True
    assert confidence_allows_apply("low") is False
    assert confidence_allows_apply("garbage") is False
