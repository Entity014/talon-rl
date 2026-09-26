"""Capacity-matched shared residual actor for the V1-B pilot."""
from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
from torch import Tensor, nn

from ...modules.v1a_actor_critic import ResidualObjectiveAdapter


class RslRlSharedResidualWrapper(nn.Module):
    def __init__(self, base: nn.Module, bottleneck_dim: int, objectives, w_ref):
        super().__init__()
        if not isinstance(base.actor, nn.Sequential) or not isinstance(base.actor[-1], nn.Linear):
            raise TypeError("base.actor must be standard rsl_rl Sequential MLP")
        self._base = base
        self.objectives = tuple(objectives)
        self.register_buffer("w_ref", torch.as_tensor(tuple(w_ref), dtype=torch.float32), persistent=True)
        if self.w_ref.numel() != len(self.objectives) or not torch.allclose(self.w_ref.sum(), torch.tensor(1.0)):
            raise ValueError("w_ref must match objectives and lie on simplex")
        self._shared_actor = nn.Sequential(*list(base.actor)[:-1])
        self._action_head = base.actor[-1]
        self.shared_residual = ResidualObjectiveAdapter(self._action_head.in_features, bottleneck_dim)

    @property
    def base(self):
        return self._base

    def actor_mean(self, obs: Tensor, w: Tensor | None = None) -> Tensor:
        h = self._shared_actor(obs)
        return self._action_head(h + self.shared_residual(h))

    def _actor_input(self, obs: Any) -> Tensor:
        if not isinstance(obs, Tensor):
            obs = self._base.get_actor_obs(obs)
            obs = self._base.actor_obs_normalizer(obs)
        return obs

    def _critic_input(self, obs: Any) -> Tensor:
        if not isinstance(obs, Tensor):
            obs = self._base.get_critic_obs(obs)
            obs = self._base.critic_obs_normalizer(obs)
        return obs

    def _distribution_from_mean(self, mean: Tensor):
        current = getattr(self._base, "distribution", None)
        distribution_type = type(current) if current is not None else torch.distributions.Normal
        if getattr(self._base, "noise_std_type", "scalar") == "scalar":
            scale = self._base.std.expand_as(mean)
        elif self._base.noise_std_type == "log":
            scale = torch.exp(self._base.log_std).expand_as(mean)
        else:
            raise ValueError("unsupported noise_std_type")
        return distribution_type(mean, scale)

    def update_distribution(self, obs: Any, w: Tensor | None = None):
        self._base.distribution = self._distribution_from_mean(self.actor_mean(self._actor_input(obs), w))
        return self._base.distribution

    def act(self, obs: Any, w: Tensor | None = None, **kwargs):
        return self.update_distribution(obs, w).sample()

    def act_inference(self, obs: Any, w: Tensor | None = None):
        return self.actor_mean(self._actor_input(obs), w)

    def get_actions_log_prob(self, actions: Tensor) -> Tensor:
        return self._base.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, critic_obs: Any, **kwargs) -> Tensor:
        return self._base.critic(self._critic_input(critic_obs))

    def state_dict(self, *args, **kwargs):
        result = OrderedDict(self._base.state_dict(*args, **kwargs))
        result.update((f"shared_residual.{k}", v) for k, v in self.shared_residual.state_dict(*args, **kwargs).items())
        result["w_ref"] = self.w_ref.detach().clone()
        return result

    def load_state_dict(self, state_dict, strict=True, assign=False):
        base_state = OrderedDict((k, v) for k, v in state_dict.items() if not k.startswith("shared_residual.") and k != "w_ref")
        residual_state = OrderedDict((k.removeprefix("shared_residual."), v) for k, v in state_dict.items() if k.startswith("shared_residual."))
        if "w_ref" in state_dict:
            self.w_ref.copy_(state_dict["w_ref"].to(self.w_ref))
        base_result = self._base.load_state_dict(base_state, strict=strict)
        residual_result = self.shared_residual.load_state_dict(residual_state, strict=strict)
        missing = (list(base_result.missing_keys) if hasattr(base_result, "missing_keys") else []) + list(residual_result.missing_keys)
        unexpected = (list(base_result.unexpected_keys) if hasattr(base_result, "unexpected_keys") else []) + list(residual_result.unexpected_keys)
        return nn.modules.module._IncompatibleKeys(missing, unexpected)
