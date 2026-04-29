"""Raw PyTorch integration — the `lensml.watch()` one-line API.

Plan refs:
    §1 — `lensml.watch(model, optimizer, dataloader)` installs hooks, streams
         metrics to local FastAPI server, monitors for pathologies.
    §8 Day 5 — implementation. Monkey-patch `optimizer.step` for pre-step
               gradient inspection. RemovableHandle teardown via context
               manager or explicit `lensml.unwatch()`.
    §8 Day 13 — checkpoint resume: `lensml.watch()` checks `LENSML_RESUME`
                env var on init.
"""

from __future__ import annotations

from typing import Any


class WatchHandle:
    """Returned by `lensml.watch()`; supports `.unwatch()` and context manager."""

    def __enter__(self) -> WatchHandle:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.unwatch()

    def unwatch(self) -> None:
        raise NotImplementedError("Implemented Day 5 — see lensml-plan.md §8")


def watch(
    model: Any,
    optimizer: Any,
    dataloader: Any,
    *,
    run_id: str | None = None,
    config: dict | None = None,
) -> WatchHandle:
    """Install lensml hooks on a PyTorch training loop.

    See plan §1 for the full one-paragraph behavior description.
    """
    raise NotImplementedError("Implemented Day 5 — see lensml-plan.md §8")


def unwatch(handle: WatchHandle) -> None:
    """Tear down a previously-installed watch."""
    handle.unwatch()
