"""Small integration boundary between rsl_rl PPO and the V1-A wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch

from .v1a_wrapper import RslRlV1AWrapper


def attach_v1a_policy(
    algorithm: Any,
    checkpoint: str | Path,
    bottleneck_dim: int,
    w_ref: Sequence[float] = (0.2,) * 5,
    objectives: Sequence[str] | None = None,
) -> RslRlV1AWrapper:
    """Load M0.1 into the existing scalar PPO policy and add V1-A adapters.

    The algorithm object is expected to be an rsl_rl PPO instance.  Its PPO
    implementation, rollout storage, critic, and distribution remain the
    existing objects; only ``algorithm.policy`` and its optimizer parameter
    set are extended.
    """
    if not hasattr(algorithm, "policy") or not hasattr(algorithm, "optimizer"):
        raise TypeError("algorithm must expose rsl_rl policy and optimizer")
    state = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    raw_model_state = state.get("model_state_dict", state)
    base = algorithm.policy
    base.load_state_dict(raw_model_state)
    wrapper_kwargs = {"bottleneck_dim": bottleneck_dim, "w_ref": w_ref}
    if objectives is not None:
        wrapper_kwargs["objectives"] = tuple(objectives)
    wrapper = RslRlV1AWrapper(base, **wrapper_kwargs)
    wrapper.to(next(base.parameters()).device)

    old_optimizer = algorithm.optimizer
    defaults = dict(old_optimizer.defaults)
    new_optimizer = type(old_optimizer)(wrapper.parameters(), **defaults)
    optimizer_state = state.get("optimizer_state_dict")
    if optimizer_state is not None:
        # rsl_rl's checkpoint parameter IDs follow the base model order.  The
        # wrapper exposes that exact order first, then newly-created adapter
        # parameters, so copy moments only for the inherited base parameters.
        new_state = new_optimizer.state_dict()
        old_states = optimizer_state.get("state", {})
        new_ids = new_state["param_groups"][0]["params"]
        old_ids = optimizer_state.get("param_groups", [{}])[0].get("params", [])
        for old_id, new_id in zip(old_ids, new_ids):
            if old_id in old_states:
                new_state["state"][new_id] = old_states[old_id]
        for key in ("betas", "eps", "weight_decay", "amsgrad", "maximize", "foreach", "capturable", "differentiable", "fused", "decoupled_weight_decay"):
            if key in optimizer_state.get("param_groups", [{}])[0]:
                new_state["param_groups"][0][key] = optimizer_state["param_groups"][0][key]
        new_state["param_groups"][0]["lr"] = optimizer_state.get("param_groups", [{}])[0].get("lr", defaults.get("lr", 1e-3))
        new_optimizer.load_state_dict(new_state)
    algorithm.policy = wrapper
    algorithm.optimizer = new_optimizer
    return wrapper
