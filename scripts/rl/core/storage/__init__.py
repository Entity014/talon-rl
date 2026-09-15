"""Advantage-estimation utilities for RL agents."""

from .rollout_storage import gae_per_objective

__all__ = ["gae_per_objective"]
