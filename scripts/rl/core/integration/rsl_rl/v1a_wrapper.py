"""Function-preserving V1-A wrapper for an existing rsl_rl ActorCritic.

This module intentionally does not import rsl_rl.  The real Isaac runtime is
expected to provide the base object.  The wrapper uses the public behavior
used by rsl_rl's on-policy runner: ``actor``, ``critic``, ``action_std``,
``distribution``, and the distribution methods.

The base object is kept as the source of truth.  Its actor/critic/std
parameters remain under their original state-dict names; only the new
parameters are registered under ``v1a_adapters.*``.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, Sequence
from typing import Any

import torch
from torch import Tensor, nn

from ...modules.v1a_actor_critic import OBJECTIVES, ResidualObjectiveAdapter


class RslRlV1AWrapper(nn.Module):
    """Wrap a live rsl_rl ActorCritic without converting its checkpoint."""

    def __init__(
        self,
        base: nn.Module,
        bottleneck_dim: int,
        objectives: Sequence[str] = OBJECTIVES,
        w_ref: Sequence[float] = (0.2,) * len(OBJECTIVES),
    ):
        super().__init__()
        objectives = tuple(objectives)
        if not objectives or len(set(objectives)) != len(objectives):
            raise ValueError("objectives must be a non-empty sequence of unique names")
        if not hasattr(base, "actor") or not hasattr(base, "critic"):
            raise TypeError("base must expose rsl_rl actor and critic")
        if not hasattr(base, "std") and not hasattr(base, "log_std"):
            raise TypeError("base must expose rsl_rl std or log_std state")
        if not isinstance(base.actor, nn.Sequential) or len(base.actor) < 2:
            raise TypeError("base.actor must be the standard rsl_rl Sequential MLP")
        if not isinstance(base.actor[-1], nn.Linear):
            raise TypeError("V1-A insertion requires the final actor module to be Linear")
        self._base = base
        self.is_recurrent = bool(getattr(base, "is_recurrent", False))
        self.objectives = objectives
        self.bottleneck_dim = bottleneck_dim
        reference = torch.as_tensor(tuple(w_ref), dtype=torch.float32)
        self._validate_w(reference, 1, len(self.objectives))
        self.register_buffer("w_ref", reference, persistent=True)
        # Use iteration, not ``children()``: rsl_rl's MLP reuses the same
        # activation module instance for every hidden layer, and PyTorch's
        # ``named_children`` de-duplicates that shared object.  Dropping it
        # would silently remove hidden activations and destroy equivalence.
        self._shared_actor = nn.Sequential(*list(base.actor)[:-1])
        self._action_head = base.actor[-1]
        hidden_dim = self._action_head.in_features
        self.v1a_adapters = nn.ModuleList(
            ResidualObjectiveAdapter(hidden_dim, bottleneck_dim) for _ in self.objectives
        )

    @property
    def base(self) -> nn.Module:
        return self._base

    @staticmethod
    def _validate_w(w: Tensor, batch_size: int, objective_count: int) -> Tensor:
        if w.ndim == 1:
            w = w.unsqueeze(0)
        if w.shape != (1, objective_count) and w.shape != (batch_size, objective_count):
            raise ValueError(f"w must have shape [1 or batch, {objective_count}]")
        if not torch.isfinite(w).all() or (w < 0).any():
            raise ValueError("w must be finite and non-negative")
        if not torch.allclose(w.sum(-1), torch.ones(w.shape[0], device=w.device, dtype=w.dtype), atol=1e-6, rtol=0):
            raise ValueError("w must lie on the probability simplex")
        return w

    def actor_mean(self, obs: Tensor, w: Tensor) -> Tensor:
        h = self._shared_actor(obs)
        weights = self._validate_w(w, h.shape[0], len(self.objectives)).to(device=h.device, dtype=h.dtype)
        if weights.shape[0] == 1 and h.shape[0] != 1:
            weights = weights.expand(h.shape[0], -1)
        residuals = torch.stack([adapter(h) for adapter in self.v1a_adapters], dim=1)
        return self._action_head(h + (weights.unsqueeze(-1) * residuals).sum(dim=1))

    def _distribution_from_mean(self, mean: Tensor) -> Any:
        """Construct the backend's existing distribution with its exact std."""
        current = getattr(self._base, "distribution", None)
        distribution_type = type(current) if current is not None else torch.distributions.Normal
        if getattr(self._base, "noise_std_type", "scalar") == "scalar":
            scale = self._base.std.expand_as(mean)
        elif self._base.noise_std_type == "log":
            scale = torch.exp(self._base.log_std).expand_as(mean)
        else:
            raise ValueError(f"unsupported rsl_rl noise_std_type: {self._base.noise_std_type}")
        return distribution_type(mean, scale)

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

    @property
    def action_mean(self) -> Tensor:
        return self._base.action_mean

    @property
    def action_std(self) -> Tensor:
        return self._base.action_std

    @property
    def entropy(self) -> Tensor:
        return self._base.entropy

    def update_distribution(self, obs: Any, w: Tensor | None = None) -> Any:
        obs = self._actor_input(obs)
        if w is None:
            w = self.w_ref
        self._base.distribution = self._distribution_from_mean(self.actor_mean(obs, w))
        return self._base.distribution

    def act(self, obs: Any, w: Tensor | None = None, **kwargs: Any) -> Tensor:
        return self.update_distribution(obs, w).sample()

    def act_inference(self, obs: Any, w: Tensor | None = None) -> Tensor:
        obs = self._actor_input(obs)
        return self.actor_mean(obs, self.w_ref if w is None else w)

    def get_actions_log_prob(self, actions: Tensor) -> Tensor:
        return self._base.distribution.log_prob(actions).sum(dim=-1)

    @property
    def entropy(self) -> Tensor:
        return self._base.distribution.entropy().sum(dim=-1)

    def evaluate(self, critic_obs: Any, **kwargs: Any) -> Tensor:
        return self._base.critic(self._critic_input(critic_obs))

    def update_normalization(self, obs: Any) -> None:
        update = getattr(self._base, "update_normalization", None)
        if update is not None:
            update(obs)

    def reset(self, dones: Tensor | None = None) -> None:
        reset = getattr(self._base, "reset", None)
        if reset is not None:
            reset(dones)

    def named_parameters(self, prefix: str = "", recurse: bool = True) -> Iterator[tuple[str, nn.Parameter]]:
        """Expose baseline names unchanged, then the V1-A extension names."""
        base_prefix = f"{prefix}." if prefix else ""
        yield from self._base.named_parameters(prefix=prefix, recurse=recurse)
        yield from self.v1a_adapters.named_parameters(prefix=f"{base_prefix}v1a_adapters", recurse=recurse)

    def state_dict(self, *args: Any, **kwargs: Any) -> OrderedDict[str, Tensor]:
        result = OrderedDict(self._base.state_dict(*args, **kwargs))
        adapter_state = self.v1a_adapters.state_dict(*args, **kwargs)
        result.update((f"v1a_adapters.{key}", value) for key, value in adapter_state.items())
        result["w_ref"] = self.w_ref.detach().clone()
        return result

    def load_state_dict(self, state_dict: Any, strict: bool = True, assign: bool = False):
        base_state = OrderedDict(
            (key, value)
            for key, value in state_dict.items()
            if not key.startswith("v1a_adapters.") and key != "w_ref"
        )
        adapter_state = OrderedDict(
            (key.removeprefix("v1a_adapters."), value)
            for key, value in state_dict.items()
            if key.startswith("v1a_adapters.")
        )
        if "w_ref" in state_dict:
            self.w_ref.copy_(state_dict["w_ref"].to(device=self.w_ref.device, dtype=self.w_ref.dtype))
        base_result = self._base.load_state_dict(base_state, strict=strict)
        adapter_result = self.v1a_adapters.load_state_dict(adapter_state, strict=strict) if adapter_state else None
        if adapter_result is None:
            return torch.nn.modules.module._IncompatibleKeys([], []) if isinstance(base_result, bool) else base_result
        base_missing = [] if isinstance(base_result, bool) else base_result.missing_keys
        base_unexpected = [] if isinstance(base_result, bool) else base_result.unexpected_keys
        return torch.nn.modules.module._IncompatibleKeys(
            base_missing + adapter_result.missing_keys,
            base_unexpected + adapter_result.unexpected_keys,
        )
