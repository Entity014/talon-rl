#!/usr/bin/env python3
"""V4-C2 twin-fidelity gate: with the same preference on both twins, how far apart do same-state twins drift over 32 steps?

Engineering check only; reads no semantic result. 2N envs, obs corruption
off (the stock noise draws differ per env and would split identical twins),
center preference for W warm-up steps, copy env j -> j+N, then both halves
keep the center preference for 32 steps. Reports per-step max |difference|
in the four semantic scores, joint positions and the objective rewards.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))  # scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from g1_evaluate import G1Evaluate, center_w, tilt_deg
from rl.core.normalization.running import RunningNormalizer
from twins import copy_state

N = 256
W = 100
STEPS = 32
TOL = 1e-4


def semantic_scores(u, a, prev):
    d = u.scene["robot"].data
    cmd = u.command_manager.get_command("base_velocity")
    return torch.stack([-((d.root_lin_vel_b[:, 0] - cmd[:, 0]).abs() + (d.root_ang_vel_b[:, 2] - cmd[:, 2]).abs()),
                        -torch.linalg.vector_norm(d.root_ang_vel_b[:, :2], dim=-1),
                        -tilt_deg(d.root_quat_w),
                        -torch.linalg.vector_norm(a - prev, dim=-1)], -1)


class TwinFidelity(G1Evaluate):
    report = "twin_fidelity.json"

    def build_env(self):
        import gymnasium as gym
        import talon_rl.tasks.locomotion.a1_env  # noqa: F401
        from talon_rl.tasks.locomotion.a1_env.v4c_env_cfg import TalonV4CEnvCfg
        cfg = TalonV4CEnvCfg(); cfg.scene.num_envs = 2 * N; cfg.seed = 0
        cfg.observations.policy.enable_corruption = False
        self.cfg = cfg
        env = gym.make(self.task, cfg=cfg)
        obs, _ = env.reset(seed=0)
        return env, obs

    def rollout(self, env, obs) -> dict:
        from talon_rl.models.authority.teacher_v4 import TeacherV4
        ck = torch.load(self.ck, map_location="cuda", weights_only=False)
        self.model = TeacherV4().cuda(); self.model.load_state_dict(ck["model"]); self.model.eval()
        self.norm = RunningNormalizer(12, center=True); self.norm.load_state_dict(ck["extrinsics_normalizer"])
        u = env.unwrapped; mgr = u.reward_manager; robot = u.scene["robot"]
        ids = torch.arange(4, device="cuda").repeat(2 * N, 1)
        w = torch.tensor(center_w(4), device="cuda").repeat(2 * N, 1)
        with torch.no_grad():
            for _ in range(W):
                obs, *_ = env.step(self.model.act_inference(obs["policy"], self.e_norm(obs), ids, w))
            src = torch.arange(N, device="cuda"); dst = src + N
            obs = copy_state(env, obs, src, dst)
            diffs = {"score": [], "joint_pos": [], "reward": []}; done_any = torch.zeros(N, dtype=torch.bool, device="cuda")
            prev = u.action_manager.action.clone()
            for _ in range(STEPS):
                a = self.model.act_inference(obs["policy"], self.e_norm(obs), ids, w)
                obs, _, te, tr, _ = env.step(a)
                done_any |= (te | tr)[src] | (te | tr)[dst]
                keep = ~done_any
                s = semantic_scores(u, a, prev); prev = a
                diffs["score"].append(float((s[src] - s[dst])[keep].abs().max()) if keep.any() else 0.0)
                diffs["joint_pos"].append(float((robot.data.joint_pos[src] - robot.data.joint_pos[dst])[keep].abs().max()) if keep.any() else 0.0)
                diffs["reward"].append(float((mgr._step_reward[src] - mgr._step_reward[dst])[keep].abs().max()) if keep.any() else 0.0)
        worst = {k: max(v) for k, v in diffs.items()}
        out = {"schema": "teacher_v4_c2_twin_fidelity_v1", "checkpoint": str(self.ck), "pairs": N, "warmup": W, "steps": STEPS,
               "obs_corruption": False, "per_step_max_abs_diff": diffs, "worst": worst,
               "pairs_terminated": int(done_any.sum()), "tolerance": TOL, "pass": bool(max(worst.values()) <= TOL)}
        self.write(out)
        print("TWIN", worst, "terminated", int(done_any.sum()), "PASS" if out["pass"] else "FAIL", flush=True)
        print("per-step score diff", [round(x, 6) for x in diffs["score"]], flush=True)
        return out


if __name__ == "__main__":
    a = TwinFidelity.parse_args((("--fold",), {"choices": ("G1-2", "G1-3"), "required": True}),
                                (("--seed",), {"type": int, "required": True}))
    TwinFidelity(a.fold, a.seed, 300, a.out).execute()
