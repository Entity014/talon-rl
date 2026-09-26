#!/usr/bin/env python3
"""The S1 regularizer scales again, this time on a real Isaac rollout.

No optimizer step is taken. The synthetic scale audit fixes the rollout; this
one draws it from the trained seed-0 policy so the losses and gradient norms
are measured against states the policy actually reaches. The temporal term
only counts consecutive pairs that did not cross a termination.
"""
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rl.core.isaac_audit import ARTIFACTS, RUNS, IsaacAudit

CHECKPOINT = "b1_p2_seed0_2026-09-21/checkpoints/update_500.pt"
NUM_ENVS, STEPS = 16, 16
COMMAND_SLICE = slice(42, 45)
SPATIAL_COEF, TEMPORAL_COEF = .1, .05
NOISE_SCALE, NOISE_CLIP = .02, .05


def grad_norm(loss, params):
    g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return float(torch.sqrt(sum((x.detach() ** 2).sum() for x in g if x is not None)))


class B1S1RealRolloutScaleAudit(IsaacAudit):
    """S1 regularizer scale on a real rollout, with no update applied."""

    root = ARTIFACTS
    run = "b1_s1_scale_audit"
    report = "real_rollout_summary.json"
    num_envs = NUM_ENVS

    def build_env(self):
        """The Talon A1 task under the B0 wrapper, nominalised, not the flat
        velocity task the other Isaac audits use."""
        import gymnasium as gym
        import talon_rl.tasks.locomotion.a1_env  # noqa: F401  (registers the task)
        from talon_rl.b0_env import B0TalonEnv
        from talon_rl.tasks.locomotion.a1_env.a1_env_cfg import IsaacLabTalonEnvCfg

        from rl.experiments.shared.train_b0 import nominalize

        cfg = IsaacLabTalonEnvCfg()
        cfg.scene.num_envs = self.num_envs
        cfg.seed = 0
        cfg.sim.dt = .01
        cfg.decimation = 1
        nominalize(cfg)
        self.base = gym.make("Isaac-Talon-A1-v0", cfg=cfg, render_mode=None).unwrapped
        env = B0TalonEnv(self.base)
        return env, env.reset()["obs"]

    def rollout(self, env, obs):
        from rl.core.modules.actor_critic import ActorCritic

        base = self.base
        m = ActorCritic(base.obs_dim, base.obs_dim, base.action_dim, 1, [64, 64]).to(base.device).eval()
        state = torch.load(RUNS / CHECKPOINT, map_location=base.device)
        m.load_state_dict(state["model"])
        m.set_scheduled_fixed_std(float(state["scheduled_std"]))

        obs_seq, act_seq, logp_seq, rew, dones = [], [], [], [], []
        for _ in range(STEPS):
            x = torch.as_tensor(obs, device=base.device, dtype=torch.float32)
            action, logp = m.act(x)
            tr = env.step(action.detach().cpu().numpy().astype(np.float32))
            obs_seq.append(x)
            act_seq.append(action.detach())
            logp_seq.append(logp.detach())
            rew.append(torch.as_tensor(tr["reward"], device=base.device))
            dones.append(torch.as_tensor(tr["done"], device=base.device))
            obs = tr["obs"]

        o = torch.stack(obs_seq)
        a = torch.stack(act_seq)
        lp = torch.stack(logp_seq)
        d = torch.stack(dones)
        flat = o.reshape(-1, base.obs_dim)
        actions = a.reshape(-1, base.action_dim)
        old = lp.reshape(-1)
        adv = torch.ones_like(old)

        mu = m.raw_mean(flat).reshape(STEPS, NUM_ENVS, -1)
        mask = torch.ones(base.obs_dim, device=base.device)
        mask[COMMAND_SLICE] = 0
        noise = torch.randn_like(o) * NOISE_SCALE
        noise.clamp_(-NOISE_CLIP, NOISE_CLIP)
        noise *= mask
        spatial = ((m.raw_mean((o + noise).reshape(-1, base.obs_dim)).reshape_as(mu) - mu) ** 2).mean()
        # only pairs that did not cross a termination are real transitions
        valid = (~d[:-1]).float()
        temporal = ((mu[1:] - mu[:-1]) ** 2).mean(dim=-1)
        temporal_loss = (temporal * valid).sum() / (valid.sum() * mu.shape[-1] + 1e-8)

        ppo = -(torch.exp(m.logp(flat, actions) - old) * adv).mean()
        actor = [p for n, p in m.named_parameters()
                 if n != "log_std" and not n.startswith("critic_")]
        weighted_s = SPATIAL_COEF * spatial
        weighted_t = TEMPORAL_COEF * temporal_loss

        result = {"rollout_shape": list(o.shape),
                  "ppo_actor_loss": float(ppo.detach()),
                  "spatial_loss": float(spatial.detach()),
                  "weighted_spatial": float(weighted_s.detach()),
                  "temporal_loss": float(temporal_loss.detach()),
                  "weighted_temporal": float(weighted_t.detach()),
                  "grad_norm_ppo": grad_norm(ppo, actor),
                  "grad_norm_spatial": grad_norm(weighted_s, actor),
                  "grad_norm_temporal": grad_norm(weighted_t, actor),
                  "valid_temporal_pairs": float(valid.sum()),
                  "temporal_pair_fraction": float(valid.mean()),
                  "mu_delta_sq_mean": float(temporal.mean()),
                  "near_fall_pairs": int(d[:-1].sum().item()),
                  "scheduled_std": float(state["scheduled_std"]),
                  "checkpoint": "seed0/update500", "no_optimizer_step": True}
        self.write(result)
        self.write({"status": "RUN_DONE", "exit_code": 0, "unix": time.time()},
                   "real_RUN_DONE.json")
        print(self.out / self.report)
        return result

    def execute(self):
        self.write({"status": "RUN_STARTED", "mode": "real_rollout_no_update",
                    "envs": NUM_ENVS, "steps": STEPS}, "real_RUN_STARTED.json")
        self.base = None
        try:
            return super().execute()
        except BaseException as e:
            self.write({"status": "ERROR", "error": str(e),
                        "traceback": traceback.format_exc()}, "real_ERROR.json")
            raise
        finally:
            if self.base is not None:
                self.base.close()


if __name__ == "__main__":
    B1S1RealRolloutScaleAudit.main()
