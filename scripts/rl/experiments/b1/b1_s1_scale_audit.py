#!/usr/bin/env python3
"""How large would the proposed S1 regularizers be against the PPO actor loss?

Read-only: it builds each candidate loss on one synthetic rollout and reports
its value and gradient norm beside PPO's, so the weights can be judged on
scale. It trains nothing and tunes nothing. Command dimensions are held fixed
in the spatial perturbation, since a regularizer must not penalise responding
to the command.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rl.core.offline_audit import ARTIFACTS, OfflineAudit

SEED = 21
OBS_DIM, ACT_DIM = 51, 12
LANES, STEPS = 8, 4
COMMAND_SLICE = slice(42, 45)
SPATIAL_COEF, TEMPORAL_COEF = .1, .05
NOISE_SCALE, NOISE_CLIP = .02, .05


def grad_norm(loss, params):
    g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return float(torch.sqrt(sum((x.detach() ** 2).sum() for x in g if x is not None)))


class B1S1ScaleAudit(OfflineAudit):
    """Scale of the proposed S1 regularizers against the PPO actor loss."""

    root = ARTIFACTS
    run = "b1_s1_scale_audit"
    report = "summary.json"

    def analyze(self):
        from rl.core.modules.actor_critic import ActorCritic

        torch.manual_seed(SEED)
        m = ActorCritic(OBS_DIM, OBS_DIM, ACT_DIM, 1, [64, 64])
        obs = torch.randn(LANES, STEPS, OBS_DIM)
        obs[:, :, COMMAND_SLICE] = torch.tensor([.5, 0., 0.])
        flat = obs.reshape(-1, OBS_DIM)

        act, old = m.act(flat)
        adv = torch.randn(flat.shape[0])
        # the original drew returns here too and never used them; the draw has
        # to stay, or every later randn comes off a different RNG state
        torch.randn(flat.shape[0])
        ratio = torch.exp(m.logp(flat, act) - old)
        ppo = -(ratio * adv).mean()

        mu = m.raw_mean(flat).reshape(LANES, STEPS, ACT_DIM)
        mask = torch.ones(OBS_DIM)
        mask[COMMAND_SLICE] = 0
        noise = torch.randn_like(obs) * NOISE_SCALE
        noise.clamp_(-NOISE_CLIP, NOISE_CLIP)
        noise *= mask
        spatial = ((m.raw_mean((obs + noise).reshape(-1, OBS_DIM)).reshape(LANES, STEPS, ACT_DIM)
                    - mu) ** 2).mean()
        temporal = ((mu[:, 1:] - mu[:, :-1]) ** 2).mean()

        actor = [p for n, p in m.named_parameters()
                 if n != "log_std" and not n.startswith("critic_")]
        return {"rollout_shape": list(obs.shape),
                "ppo_actor_loss": float(ppo.detach()),
                "spatial_loss": float(spatial.detach()),
                "weighted_spatial": float(SPATIAL_COEF * spatial.detach()),
                "temporal_loss": float(temporal.detach()),
                "weighted_temporal": float(TEMPORAL_COEF * temporal.detach()),
                "grad_norm_ppo": grad_norm(ppo, actor),
                "grad_norm_spatial": grad_norm(SPATIAL_COEF * spatial, actor),
                "grad_norm_temporal": grad_norm(TEMPORAL_COEF * temporal, actor),
                "command_mask_excluded_indices": list(range(COMMAND_SLICE.start,
                                                            COMMAND_SLICE.stop)),
                "read_only": True}

    def summarize(self, report):
        print(self.out / self.report)


if __name__ == "__main__":
    B1S1ScaleAudit.main()
