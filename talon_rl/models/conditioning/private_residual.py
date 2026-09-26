"""V2-K: combination-rescue actor.

Combines:
- V2-C private residual capacity, but without a softmax router;
- V2-H continuous preference-conditioned coefficient generation.

Preference w generates unconstrained coefficients c_k(w) over four private
residual modules E_k(h):

    h' = h + sum_k c_k(w) E_k(h)

The modules have no semantic labels.  Their parameters are nonzero at
initialization, while the coefficient output layer is zero-initialized.
Therefore the combined residual is exactly zero at K0 while the coefficient
mapping can receive gradients immediately.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from talon_rl.models.conditioning.single_site_film import V2BSingleSiteFiLMActorCritic
from talon_rl.models.foundations.preference_embedding import NUM_OBJECTIVES


class _PrivateResidualModule(nn.Module):
    def __init__(self, hidden_dim: int, bottleneck_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, bottleneck_dim)
        self.act = nn.ELU()
        self.fc2 = nn.Linear(bottleneck_dim, hidden_dim)

        # Nonzero private basis functions from the start.
        nn.init.orthogonal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.orthogonal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

        with torch.no_grad():
            self.fc1.weight.mul_(0.10)
            self.fc2.weight.mul_(0.10)

    def forward(self, h: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(h)))


class V2KHyperPrivateResidualActorCritic(V2BSingleSiteFiLMActorCritic):
    NUM_MODULES = 4
    MODULE_BOTTLENECK = 32
    COEFF_HIDDEN = 16

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: list[int] | None = None,
    ):
        hidden_dims = hidden_dims or [128, 128, 128]
        super().__init__(obs_dim, action_dim, hidden_dims)

        self.preference_coeff = nn.Sequential(
            nn.Linear(NUM_OBJECTIVES, self.COEFF_HIDDEN),
            nn.ELU(),
            nn.Linear(self.COEFF_HIDDEN, self.NUM_MODULES),
        )
        self.private_modules = nn.ModuleList(
            [
                _PrivateResidualModule(self.FILM_DIM, self.MODULE_BOTTLENECK)
                for _ in range(self.NUM_MODULES)
            ]
        )

        # Exact function preservation while retaining immediate coefficient
        # gradients through the already-nonzero private module outputs.
        with torch.no_grad():
            self.preference_coeff[-1].weight.zero_()
            self.preference_coeff[-1].bias.zero_()

    def module_coefficients(self, w: Tensor) -> Tensor:
        return self.preference_coeff(w)

    def private_module_outputs(self, h: Tensor) -> Tensor:
        return torch.stack([m(h) for m in self.private_modules], dim=1)

    def _combined_private_residual(self, h: Tensor, w: Tensor) -> Tensor:
        coeff = self.module_coefficients(w)
        outs = self.private_module_outputs(h)
        return torch.sum(coeff.unsqueeze(-1) * outs, dim=1)

    def _modular_hidden(self, obs: Tensor, w: Tensor) -> Tensor:
        h = self._film_hidden(obs, w)
        return h + self._combined_private_residual(h, w)

    def _actor_features_v2a(self, obs: Tensor, w: Tensor) -> Tensor:
        h = self._modular_hidden(obs, w)
        e = self.preference_embedding(w)
        return torch.cat((h, e), dim=-1)


def initialize_from_v2b(
    model: V2KHyperPrivateResidualActorCritic,
    v2b_state: dict,
) -> None:
    """Copy V2-B exactly and zero only preference coefficients."""
    target = model.state_dict()
    source = v2b_state.get("model", v2b_state.get("model_state_dict", v2b_state))

    for key in list(target):
        if key in source and target[key].shape == source[key].shape:
            target[key].copy_(source[key])

    target["preference_coeff.2.weight"].zero_()
    target["preference_coeff.2.bias"].zero_()
    model.load_state_dict(target)
