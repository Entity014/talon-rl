"""Env Factor Encoder (mu, RMA Phase 1) — compresses privileged extrinsics
e_t into a latent z_t. Lives here (trainer/algorithm layer), not in an
Isaac Lab Manager, because it must joint-train with the base policy
through the same optimizer (see scripts/rl/core/algorithms/moppo.py).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EnvFactorEncoder(nn.Module):
    def __init__(self, extrinsics_dim: int, latent_dim: int, hidden_dims: list[int] | None = None):
        super().__init__()
        # RMA's own encoder (Kumar et al. 2021, Section B "Training Details"):
        # a 3-layer MLP with hidden sizes (256, 128) encoding e_t in R^17 into
        # z_t in R^8 -- this was a single 32-unit hidden layer before
        # 2026-09-18, found smaller than the reference architecture while
        # reviewing the paper's own training-details section.
        hidden_dims = hidden_dims if hidden_dims is not None else [256, 128]
        layers: list[nn.Module] = []
        prev_dim = extrinsics_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev_dim, h), nn.ELU()]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, e_t: torch.Tensor) -> torch.Tensor:
        return self.net(e_t)
