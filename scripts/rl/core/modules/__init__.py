"""Definitions for neural-network components for RL agents."""

from .actor_critic import ActorCritic
from .v1a_actor_critic import (
    OBJECTIVES,
    V1AActorCritic,
    adapter_parameter_count,
    choose_adapter_bottleneck,
)

__all__ = ["ActorCritic", "V1AActorCritic", "OBJECTIVES", "adapter_parameter_count", "choose_adapter_bottleneck"]
