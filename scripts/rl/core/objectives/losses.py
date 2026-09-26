"""D3PO's Late-Stage Weighting (LSW) + diversity regularizer (Ambadkar et al.
2026, arXiv:2602.07764) — replaces AMOR-style early scalarization (clip the
w-scalarized advantage) with per-objective clipping, weighted-summed after
the clip instead of before. See
scripts/rl/core/algorithms/moppo.py for how MOPPOTrainer wires this in.
"""

from __future__ import annotations

import numpy as np
import torch


def normalize_per_objective(adv: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """adv: (B, K) -> (B, K), each objective column independently zero-mean
    unit-std ("non-homogeneity via per-objective advantage normalization",
    D3PO's own framing) — not a single combined-scalar normalization."""
    return ((adv - adv.mean(axis=0)) / (adv.std(axis=0) + eps)).astype(np.float32)


def d3po_actor_loss(ratio: torch.Tensor, adv: torch.Tensor, w: torch.Tensor, clip_eps: float) -> torch.Tensor:
    """ratio: (B,), adv: (B, K) per-objective (already normalized), w: (B, K)
    -> scalar loss. Clips each objective independently with the SAME ratio
    (one policy), then weights by w and sums — deferring scalarization past
    the clip, unlike AMOR's clip(ratio, w . adv)."""
    clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    per_objective = torch.min(ratio[:, None] * adv, clipped_ratio[:, None] * adv)
    weighted = (w * per_objective).sum(-1)
    return -weighted.mean()


def diversity_regularizer_loss(
    mean_w: torch.Tensor, mean_w_prime: torch.Tensor, w: torch.Tensor, w_prime: torch.Tensor,
    std: torch.Tensor, alpha: float, min_std: float = 0.5, max_loss: float = 100.0,
) -> torch.Tensor:
    """Penalizes the policy for behaving too similarly under dissimilar
    preference vectors (prevents representational collapse). mean_w/
    mean_w_prime: (B, action_dim) — the policy's deterministic mean action
    under the same state with preference w vs. w_prime. w/w_prime: (B, K).
    std: (action_dim,) — the policy's action std (state/preference-
    independent, so KL between the two diagonal Gaussians has a closed
    form: no need to sample).

    min_std floors the std used in this formula's denominator (NOT
    ActorCritic.LOG_STD_MIN, a different floor for a different purpose) --
    found 2026-09-18: re-enabling this term (diversity_lambda 0->0.05) at
    the same time the policy's actual std was sitting near its own floor
    (~0.2, ActorCritic.LOG_STD_MIN=-1.6) produced a policy loss in the
    billions within ~10 updates (2*std**2 in the denominator is ~0.08 at
    that std, so even a modest few-unit gap between mean_w and mean_w'
    blows kl up to the hundreds before it's squared again). The formula
    is the textbook closed-form KL between two diagonal Gaussians sharing
    a std, which correctly diverges as std->0 for any nonzero mean gap --
    mathematically expected, but not a signal this repo's training loop
    should ever see at this magnitude. min_std decouples "how different
    must behavior be across w" from however collapsed the policy's actual
    exploration std currently is, which is the term's actual intent (a
    behavioral-diversity requirement, not an exploration-entropy one).

    max_loss is a second, independent safety net (defense-in-depth, not a
    substitute for min_std) — caps the returned value regardless of how
    large mean_w/mean_w' ever get, so this term can never again dominate
    policy_loss by 6+ orders of magnitude the way it did before min_std
    existed, even under some future combination neither of these two
    floors was tuned against."""
    safe_std = std.clamp(min=min_std)
    kl = ((mean_w - mean_w_prime) ** 2 / (2 * safe_std**2)).sum(-1)
    l1 = (w - w_prime).abs().sum(-1)
    return ((kl - alpha * l1) ** 2).mean().clamp(max=max_loss)
