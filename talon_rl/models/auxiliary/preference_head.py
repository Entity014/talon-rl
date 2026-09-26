"""V1-C shared actor with a training-only early-hidden preference head."""
from __future__ import annotations
import torch
from torch import Tensor, nn
from talon_rl.models.foundations.three_objective import V1CSharedActorCritic, NUM_OBJECTIVES

class D4BAuxActorCritic(V1CSharedActorCritic):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list[int] | None = None):
        super().__init__(obs_dim, action_dim, hidden_dims)
        self.preference_head = nn.Linear((hidden_dims or [128])[0], NUM_OBJECTIVES)

    def actor_hidden_first(self, obs: Tensor, w: Tensor) -> Tensor:
        x = self._with_w(obs, w)
        return self.actor_body[1](self.actor_body[0](x))

    def predict_preference(self, obs: Tensor, w: Tensor) -> Tensor:
        return self.preference_head(self.actor_hidden_first(obs, w))

    def auxiliary_preference_loss(self, obs: Tensor, w: Tensor) -> Tensor:
        return (self.predict_preference(obs, w) - w).pow(2).mean()

def initialize_from_v1c(model: D4BAuxActorCritic, source_state: dict) -> None:
    target = model.state_dict()
    for key, value in source_state.items():
        if key in target and target[key].shape == value.shape:
            target[key].copy_(value)
    # The head is deliberately not part of the actor deployment function.
    nn.init.zeros_(model.preference_head.weight)
    nn.init.zeros_(model.preference_head.bias)
    model.load_state_dict(target, strict=False)
