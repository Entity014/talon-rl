"""V2-C: V2-B plus preference-gated private residual experts.

Treatment:
- retain the complete V2-B direct + embedding + single-site FiLM actor,
- add four small residual experts on the post-FiLM hidden feature,
- route them with a deterministic softmax router conditioned only on preference,
- zero-initialize each expert output layer and router logits.

At initialization the routed residual is exactly zero, so V2-C is
function-preserving with respect to a frozen V2-B checkpoint.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from talon_rl.models.conditioning.single_site_film import V2BSingleSiteFiLMActorCritic
from talon_rl.models.foundations.preference_embedding import NUM_OBJECTIVES


class _ResidualExpert(nn.Module):
    def __init__(self, hidden_dim: int, bottleneck_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, bottleneck_dim)
        self.act = nn.ELU()
        self.fc2 = nn.Linear(bottleneck_dim, hidden_dim)
        with torch.no_grad():
            self.fc2.weight.zero_()
            self.fc2.bias.zero_()

    def forward(self, h: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(h)))


class V2CPreferenceGatedResidualActorCritic(V2BSingleSiteFiLMActorCritic):
    NUM_EXPERTS = 4
    EXPERT_BOTTLENECK = 32

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: list[int] | None = None,
    ):
        hidden_dims = hidden_dims or [128, 128, 128]
        super().__init__(obs_dim, action_dim, hidden_dims)

        self.preference_router = nn.Linear(NUM_OBJECTIVES, self.NUM_EXPERTS)
        self.residual_experts = nn.ModuleList(
            [
                _ResidualExpert(self.FILM_DIM, self.EXPERT_BOTTLENECK)
                for _ in range(self.NUM_EXPERTS)
            ]
        )

        with torch.no_grad():
            self.preference_router.weight.zero_()
            self.preference_router.bias.zero_()

    def routing_weights(self, w: Tensor) -> Tensor:
        return torch.softmax(self.preference_router(w), dim=-1)

    def _modular_hidden(self, obs: Tensor, w: Tensor) -> Tensor:
        h = self._film_hidden(obs, w)
        gates = self.routing_weights(w)
        expert_outputs = torch.stack([expert(h) for expert in self.residual_experts], dim=1)
        residual = torch.sum(gates.unsqueeze(-1) * expert_outputs, dim=1)
        return h + residual

    def _actor_features_v2a(self, obs: Tensor, w: Tensor) -> Tensor:
        h = self._modular_hidden(obs, w)
        e = self.preference_embedding(w)
        return torch.cat((h, e), dim=-1)


def initialize_from_v2b(
    model: V2CPreferenceGatedResidualActorCritic,
    v2b_state: dict,
) -> None:
    """Copy V2-B exactly and keep the modular treatment function-preserving."""
    target = model.state_dict()
    source = v2b_state.get("model", v2b_state.get("model_state_dict", v2b_state))

    for key in list(target):
        if key in source and target[key].shape == source[key].shape:
            target[key].copy_(source[key])

    target["preference_router.weight"].zero_()
    target["preference_router.bias"].zero_()

    for i in range(model.NUM_EXPERTS):
        target[f"residual_experts.{i}.fc2.weight"].zero_()
        target[f"residual_experts.{i}.fc2.bias"].zero_()

    model.load_state_dict(target)
