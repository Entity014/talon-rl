"""Guarded inference runtime for exported TALON policies.

This module stops at the policy boundary: hardware-specific sensor transport,
real-time scheduling, and emergency-stop handling remain the responsibility of
the robot adapter that calls it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class PolicyRuntimeConfig:
    """Shape and action limits required by a deployment adapter."""

    observation_dim: int
    action_dim: int
    action_low: float = -1.0
    action_high: float = 1.0

    def __post_init__(self) -> None:
        if self.observation_dim <= 0 or self.action_dim <= 0:
            raise ValueError("observation_dim and action_dim must be positive")
        if not self.action_low < self.action_high:
            raise ValueError("action_low must be less than action_high")


class PolicyRuntime:
    """Load and guard a TorchScript actor at the deployment boundary."""

    def __init__(self, policy_path: str, config: PolicyRuntimeConfig):
        self.config = config
        self.policy = torch.jit.load(policy_path, map_location="cpu")
        self.policy.eval()

    def act(self, observation: np.ndarray) -> np.ndarray:
        """Return a finite, bounded batch of actions.

        Rejecting invalid observations is safer than allowing NaNs to reach a
        motor adapter; actions are clipped because the final safety bound must
        live immediately before any hardware command is issued.
        """
        observation = np.asarray(observation, dtype=np.float32)
        if observation.ndim != 2:
            raise ValueError(f"observation must have shape (batch, {self.config.observation_dim})")
        if observation.shape[1] != self.config.observation_dim:
            raise ValueError(
                f"expected observation width {self.config.observation_dim}, got {observation.shape[1]}"
            )
        if not np.all(np.isfinite(observation)):
            raise ValueError("observation contains non-finite values")

        with torch.inference_mode():
            action = self.policy(torch.from_numpy(observation))
        if not isinstance(action, torch.Tensor):
            raise TypeError("exported policy must return a torch.Tensor")
        if action.ndim != 2 or tuple(action.shape) != (observation.shape[0], self.config.action_dim):
            raise ValueError(
                f"expected action shape ({observation.shape[0]}, {self.config.action_dim}), got {tuple(action.shape)}"
            )
        action_np = action.detach().cpu().numpy()
        if not np.all(np.isfinite(action_np)):
            raise ValueError("policy returned non-finite actions")
        return np.clip(action_np, self.config.action_low, self.config.action_high).astype(np.float32)
