"""Checkpoint service interface."""
from __future__ import annotations
from typing import Any, Protocol, runtime_checkable

@runtime_checkable
class CheckpointManager(Protocol):
    def save(self, path: str, state: dict[str, Any]) -> None: ...
    def load(self, path: str) -> dict[str, Any]: ...
