#!/usr/bin/env python3
"""V4-C2 null-only branch feasibility: how different are two branches restored from the same snapshot under the SAME preference, per horizon?

No preference effect is measured here. Per checkpoint and env seed: 256 envs
run the center preference (obs corruption off) for W steps; each env's state
is one snapshot (the statistical unit). Three branches are restored from the
snapshot in sequence, all with the center preference, each rolled 32 steps.
For H in {1,2,4,8,16,32} and each semantic score S_i (T, A, O, S; higher is
better), per snapshot: S_i averaged over steps 1..H of a branch, and
null deltas between branch positions (2-1, 3-2, 3-1). Reported per H:
mean, median, std, MAD, bootstrap 95% CI of the mean over snapshots,
p95/p99 of |delta|, the branch-order effect (mean of each position delta),
and dependence on command speed (split at the median).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))  # scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from g1_evaluate import G1Evaluate, center_w
from rl.core.normalization.running import RunningNormalizer

from twins import restore, semantic_scores, snapshot

N = 256
W = 100
STEPS = 32
HS = (1, 2, 4, 8, 16, 32)
LABELS = ("T", "A", "O", "S")
ENV_SEEDS = (0, 1)
BOOT = 2000


def boot_ci(x, rng):
    idx = rng.integers(0, len(x), (BOOT, len(x)))
    m = x[idx].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def summarize(d, rng):
    ad = np.abs(d)
    return {"n": int(len(d)), "mean": float(d.mean()), "median": float(np.median(d)), "std": float(d.std()),
            "mad": float(np.median(np.abs(d - np.median(d)))), "ci95_mean": boot_ci(d, rng),
            "p95_abs": float(np.percentile(ad, 95)), "p99_abs": float(np.percentile(ad, 99))}


class NullBranch(G1Evaluate):
    report = "null_branch_feasibility.json"

    def build_env(self):
        import gymnasium as gym
        import talon_rl.tasks.locomotion.a1_env  # noqa: F401
        from talon_rl.tasks.locomotion.a1_env.v4c_env_cfg import TalonV4CEnvCfg
        cfg = TalonV4CEnvCfg(); cfg.scene.num_envs = N; cfg.seed = 0
        cfg.observations.policy.enable_corruption = False
        self.cfg = cfg
        env = gym.make(self.task, cfg=cfg)
        obs, _ = env.reset(seed=0)
        return env, obs

    def branch(self, env, snap, ids, w):
        u = env.unwrapped
        obs = restore(env, snap, settle=0)
        prev = snap["action"].clone()
        S, D = [], []
        with torch.no_grad():
            for _ in range(STEPS):
                a = self.model.act_inference(obs["policy"], self.e_norm(obs), ids, w)
                obs, _, te, tr, _ = env.step(a)
                S.append(semantic_scores(u, a, prev)); prev = a
                D.append(te | tr)
        S = torch.stack(S).cpu().numpy(); D = torch.stack(D).cpu().numpy()
        alive = ~np.cumsum(D, 0).astype(bool)  # [STEPS, N]: no episode end up to and including step t
        return S, alive

    def rollout(self, env, obs) -> dict:
        from talon_rl.models.authority.teacher_v4 import TeacherV4
        ck = torch.load(self.ck, map_location="cuda", weights_only=False)
        self.model = TeacherV4().cuda(); self.model.load_state_dict(ck["model"]); self.model.eval()
        self.norm = RunningNormalizer(12, center=True); self.norm.load_state_dict(ck["extrinsics_normalizer"])
        u = env.unwrapped
        ids = torch.arange(4, device="cuda").repeat(N, 1)
        w = torch.tensor(center_w(4), device="cuda").repeat(N, 1)
        rng = np.random.default_rng(0)
        per_H = {H: {"21": [], "32": [], "31": [], "speed": []} for H in HS}
        excluded = {H: 0 for H in HS}
        for es in ENV_SEEDS:
            obs, _ = env.reset(seed=es)
            with torch.no_grad():
                for _ in range(W):
                    obs, *_ = env.step(self.model.act_inference(obs["policy"], self.e_norm(obs), ids, w))
            snap = snapshot(env, obs)
            speed = torch.linalg.vector_norm(u.command_manager.get_command("base_velocity")[:, :2], dim=-1).cpu().numpy()
            b = [self.branch(env, snap, ids, w) for _ in range(3)]
            for H in HS:
                ok = b[0][1][H - 1] & b[1][1][H - 1] & b[2][1][H - 1]
                excluded[H] += int((~ok).sum())
                m = [x[0][:H].mean(0)[ok] for x in b]  # [n_ok, 4] per branch
                per_H[H]["21"].append(m[1] - m[0]); per_H[H]["32"].append(m[2] - m[1]); per_H[H]["31"].append(m[2] - m[0])
                per_H[H]["speed"].append(speed[ok])
        out = {"schema": "teacher_v4_c2_null_branch_v1", "checkpoint": str(self.ck), "snapshots_per_seed": N, "env_seeds": list(ENV_SEEDS),
               "warmup": W, "obs_corruption": False, "restore_settle": 0, "horizons": {}}
        for H in HS:
            d21 = np.concatenate(per_H[H]["21"]); d32 = np.concatenate(per_H[H]["32"]); d31 = np.concatenate(per_H[H]["31"])
            sp = np.concatenate(per_H[H]["speed"]); fast = sp > np.median(sp)
            row = {"excluded_terminated": excluded[H]}
            for i, lab in enumerate(LABELS):
                row[lab] = {"null_21": summarize(d21[:, i], rng),
                            "order_effect_means": {"pos2-pos1": float(d21[:, i].mean()), "pos3-pos2": float(d32[:, i].mean()), "pos3-pos1": float(d31[:, i].mean())},
                            "abs_null_by_speed": {"slow": float(np.abs(d21[~fast, i]).mean()), "fast": float(np.abs(d21[fast, i]).mean())}}
            out["horizons"][str(H)] = row
            print("H", H, {lab: (round(row[lab]["null_21"]["std"], 5), round(row[lab]["null_21"]["p95_abs"], 5)) for lab in LABELS}, "excl", excluded[H], flush=True)
        self.write(out)
        return out


if __name__ == "__main__":
    a = NullBranch.parse_args((("--fold",), {"choices": ("G1-2", "G1-3"), "required": True}),
                              (("--seed",), {"type": int, "required": True}))
    NullBranch(a.fold, a.seed, 300, a.out).execute()
