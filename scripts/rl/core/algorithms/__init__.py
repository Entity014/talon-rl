"""Training algorithm families and their stable trainer contract."""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .moppo import MOPPOConfig, MOPPOTrainer


@runtime_checkable
class Trainer(Protocol):
    """Small public surface expected by entry points and experiment drivers."""

    def update(self) -> dict[str, Any]: ...
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...


__all__ = ["Trainer", "MOPPOConfig", "MOPPOTrainer"]
