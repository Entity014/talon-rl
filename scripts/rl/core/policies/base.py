"""Stable policy interfaces for RL implementations."""
from __future__ import annotations
from typing import Protocol, runtime_checkable
from torch import Tensor

@runtime_checkable
class Policy(Protocol):
    """Minimal policy contract consumed by trainers and rollout collectors."""
    def act(self, obs: Tensor, context: Tensor | None = None) -> Tensor: ...
    def act_inference(self, obs: Tensor, context: Tensor | None = None) -> Tensor: ...
    def value(self, obs: Tensor, context: Tensor | None = None) -> Tensor: ...

@runtime_checkable
class ActionEvaluator(Protocol):
    """Optional extension used by PPO-like objectives."""
    def logp(self, obs: Tensor, action: Tensor, context: Tensor | None = None) -> Tensor: ...
    def entropy(self, obs: Tensor, context: Tensor | None = None) -> Tensor: ...
