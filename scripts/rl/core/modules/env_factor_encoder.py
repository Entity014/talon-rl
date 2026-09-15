"""Env Factor Encoder (mu, RMA Phase 1) — compresses privileged extrinsics
e_t into a latent z_t. Lives here (trainer/algorithm layer), not in an
Isaac Lab Manager, because it must joint-train with the base policy
through the same optimizer (see scripts/rl/core/algorithms/moppo.py).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EnvFactorEncoder(nn.Module):
    def __init__(self, extrinsics_dim: int, latent_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(extrinsics_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, e_t: torch.Tensor) -> torch.Tensor:
        return self.net(e_t)
