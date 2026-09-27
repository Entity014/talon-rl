#!/usr/bin/env python3
"""T3-B objective-divisor protocol, re-run on a chosen env: abs-mean of each raw T/A/O/S objective under the M0 root policy.

Same protocol as post_v2_t3b_scaling_audit.py (tag pre-reorg-2026-09-26),
whose abs_mean candidate became NORMALIZATION_DIVISORS: M0 model_299 loaded
into V1CSharedActorCritic, stochastic actions clamped to +-1, w=[1,0,0],
16 envs x 192 steps x reset seeds 230001/230101/230201, objectives summed
from reward_manager._step_reward. `--env stock` is the control: it must
reproduce the frozen divisors (on 2026-09-27 it did, bit for bit).
`--env v4c` measures the V4-C divisors.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))  # scripts/

import numpy as np
import torch

from rl.core.diagnostics.isaac_audit import REPO, IsaacAudit, obs_tensor

CHECKPOINT = REPO / "runs/m0_1_seed0_2026-09-22/model_299.pt"
RESET_SEEDS = (230001, 230101, 230201)
STEPS = 192
REPRO_RTOL = 0.05


class DivisorCalibration(IsaacAudit):
    """T3-B divisor protocol on one env."""
    run = "teacher_v4_c0a3_divisor_calibration-2026-09-27"
    num_envs = 16

    def __init__(self, out, env_name: str):
        super().__init__(out)
        self.env_name = env_name
        self.report = f"{env_name}.json"
        self.seed = RESET_SEEDS[0]

    def build_env(self):
        import gymnasium as gym
        if self.env_name == "stock":
            import isaaclab_tasks  # noqa: F401
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            # T3-B did not override the USD path, so neither does the control.
            cfg, task = UnitreeA1FlatEnvCfg(), "Isaac-Velocity-Flat-Unitree-A1-v0"
        elif self.env_name == "v4c":
            import talon_rl.tasks.locomotion.a1_env  # noqa: F401
            from talon_rl.tasks.locomotion.a1_env.v4c_env_cfg import TalonV4CEnvCfg
            cfg, task = TalonV4CEnvCfg(), "Isaac-Talon-A1-V4C-v0"
        else:  # v4c_s1: adds the action-jerk term; its abs-mean is the S1 divisor
            import talon_rl.tasks.locomotion.a1_env  # noqa: F401
            from talon_rl.tasks.locomotion.a1_env.v4c_env_cfg import TalonV4CS1EnvCfg
            cfg, task = TalonV4CS1EnvCfg(), "Isaac-Talon-A1-V4C-S1-v0"
        cfg.scene.num_envs = self.num_envs
        cfg.seed = self.seed
        self.cfg = cfg
        env = gym.make(task, cfg=cfg)
        obs, _ = env.reset(seed=self.seed)
        return env, obs_tensor(obs).cuda()

    def rollout(self, env, obs) -> dict:
        from talon_rl.models.foundations.three_objective import V1CSharedActorCritic, initialize_from_rsl_m01
        from talon_rl.rewards.objectives import NORMALIZATION_DIVISORS, OBJECTIVE_ORDER, raw_objective_vector

        torch.manual_seed(self.seed)
        ad = env.unwrapped.action_manager.total_action_dim
        model = V1CSharedActorCritic(obs.shape[-1], ad).cuda()
        initialize_from_rsl_m01(model, CHECKPOINT, device="cpu")
        model.eval()
        w = torch.tensor([1., 0., 0.], device="cuda").repeat(self.num_envs, 1)
        mgr = env.unwrapped.reward_manager
        rows, seed_rows, jerk_rows = [], [], []
        for seed in RESET_SEEDS:
            cur, _ = env.reset(seed=seed)
            cur = obs_tensor(cur).cuda()
            R = []; J = []
            for _ in range(STEPS):
                with torch.no_grad():
                    a, _ = model.act_with_preference(cur, w)
                nxt, *_ = env.step(torch.clamp(a, -1, 1))
                raw = mgr._step_reward.detach().cpu().numpy().astype(np.float64)
                terms = {n: raw[:, i] for i, n in enumerate(mgr.active_terms)}
                R.append(raw_objective_vector(terms, shape=(self.num_envs,)))
                if "action_jerk_l2" in terms:
                    J.append(terms["action_jerk_l2"])
                cur = obs_tensor(nxt).cuda()
            R = np.asarray(R).reshape(-1, 4)
            rows.append(R)
            seed_rows.append({"seed": seed, "abs_mean": np.abs(R).mean(0).tolist()})
            if J:
                jerk_rows.append(np.asarray(J).reshape(-1))
        A = np.concatenate(rows)
        abs_mean = np.abs(A).mean(0)
        frozen = np.asarray(NORMALIZATION_DIVISORS, np.float64)
        rel = (abs_mean - frozen) / frozen
        out = {
            "env": self.env_name, "objective_order": list(OBJECTIVE_ORDER),
            "protocol": {"num_envs": self.num_envs, "steps": STEPS, "reset_seeds": list(RESET_SEEDS),
                         "checkpoint": str(CHECKPOINT.relative_to(REPO)), "checkpoint_sha256": self.sha(CHECKPOINT),
                         "torch_seed": self.seed},
            "abs_mean": abs_mean.tolist(), "mean": A.mean(0).tolist(), "std": A.std(0).tolist(),
            "seed_rows": seed_rows,
            "frozen_t3b_divisors": frozen.tolist(), "relative_diff_vs_frozen": rel.tolist(),
        }
        if jerk_rows:
            out["s1_action_jerk_abs_mean"] = float(np.abs(np.concatenate(jerk_rows)).mean())
        if self.env_name == "stock":
            out["checks"] = {"reproduces_frozen_divisors": bool(np.all(np.abs(rel) < REPRO_RTOL))}
            out["pass"] = all(out["checks"].values())
        self.write(out)
        print({k: out[k] for k in ("env", "abs_mean", "relative_diff_vs_frozen")}, out.get("checks", ""), flush=True)
        if out.get("pass") is False:
            raise SystemExit(1)
        return out


if __name__ == "__main__":
    a = DivisorCalibration.parse_args((("--env",), {"choices": ("stock", "v4c", "v4c_s1"), "required": True}))
    DivisorCalibration(a.out, a.env).execute()
