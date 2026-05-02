"""Confidence calibration via self-consistency.

Plan §5.4:
    Sample the diagnosis call N=3 times at temperature 0.7. ≥2/3 agree
    on the pathology label → MEDIUM. 3/3 → HIGH. <2/3 → LOW; present as
    "I noticed X but I'm not sure" without a fix proposal.

    Hard rule: never apply a fix below MEDIUM without explicit user
    override.
"""

from __future__ import annotations

from collections import Counter
from typing import Literal

ConfidenceLevel = Literal["low", "medium", "high"]


def consensus(labels: list[str | None]) -> tuple[str | None, ConfidenceLevel]:
    """Vote on a list of candidate labels.

    Args:
        labels: one entry per sample. ``None`` entries (e.g. failed parses)
            are dropped before voting.

    Returns:
        (winning_label, confidence). Returns ``(None, "low")`` if the
        majority threshold isn't met or no labels are valid.
    """
    valid = [v for v in labels if v is not None and v != ""]
    if not valid:
        return None, "low"

    counts = Counter(valid)
    most_common, count = counts.most_common(1)[0]
    n_total = len(valid)

    if count == n_total and n_total >= 2:
        return most_common, "high"
    if count * 2 > n_total:
        return most_common, "medium"
    return None, "low"


def confidence_allows_apply(level: str) -> bool:
    """Plan §5.4: never apply a fix below MEDIUM."""
    return level in ("medium", "high")
