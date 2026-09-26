#!/usr/bin/env python3
"""How far the GAE target sits from the Monte-Carlo return it approximates.

Two audits run this over different pairs of training arms: one comparing
critic-update counts, one comparing GAE lambda. Each arm is rolled fresh from
its own snapshots, and the gap is reported per objective head.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rl.core.diagnostics.isaac_audit import RUNS, IsaacAudit, obs_tensor

ORDER = ("T", "A", "O", "S")
SNAPS = (10, 25, 50)
GAMMA = .99
STEPS = 64
HEADS = 4
PREFS = {"T": np.array([.7, .1, .1, .1], np.float32),
         "A": np.array([.1, .7, .1, .1], np.float32),
         "O": np.array([.1, .1, .7, .1], np.float32),
         "S": np.array([.1, .1, .1, .7], np.float32)}


def terms(raw, names):
    return {n: raw[:, i] for i, n in enumerate(names)}


def monte_carlo(r, done):
    out = np.zeros_like(r)
    run = np.zeros_like(r[0])
    for t in range(len(r) - 1, -1, -1):
        run = r[t] + GAMMA * run * (~done[t])[:, None]
        out[t] = run
    return out


class TargetGapAudit(IsaacAudit):
    """GAE target against the Monte-Carlo return, per arm and snapshot."""

    report = "target_gap_audit.json"
    num_envs = 32
    set_usd_path = False       # these audits ran against Isaac's own A1 asset
    arms: dict = {}            # tag -> (run directory, lambda or None for the default)
    seed_base: int = 0

    def arm_snapshot(self, env, obs_dim, run, lam, lab, bi, snap):
        from talon_rl.rewards.objectives import normalized_objective_vector
        from talon_rl.models.foundations.four_objective import T4SharedActorCritic, vector_gae

        u = env.unwrapped
        mgr = u.reward_manager
        n = self.num_envs
        m = T4SharedActorCritic(obs_dim, u.action_manager.total_action_dim).cuda()
        m.load_state_dict(torch.load(RUNS / run / f"{lab}_snap_{snap}.pt",
                                     map_location="cuda", weights_only=False)["model"])
        m.eval()
        w = torch.tensor(PREFS[lab], device="cuda").repeat(n, 1)
        cur, _ = env.reset(seed=self.seed_base + bi * 1000 + snap)
        cur = obs_tensor(cur).cuda()
        r, d, v = [], [], []
        with torch.no_grad():
            for _ in range(STEPS):
                v.append(m.value_with_preference(cur, w).cpu().numpy())
                a = m.act_inference_with_preference(cur, w)
                nxt, _, te, tr, _ = env.step(a)
                raw = mgr._step_reward.detach().cpu().numpy()
                r.append(normalized_objective_vector(terms(raw, list(mgr.active_terms)),
                                                     shape=(n,)) * u.step_dt)
                d.append((te | tr).cpu().numpy())
                cur = obs_tensor(nxt).cuda()
            nv = m.value_with_preference(cur, w)

        r = np.asarray(r)
        d = np.asarray(d, bool)
        v = np.asarray(v)
        mc = monte_carlo(r, d)
        rt = torch.tensor(r, dtype=torch.float32, device="cuda")
        vt = torch.tensor(v, dtype=torch.float32, device="cuda")
        dt = torch.tensor(d, device="cuda")
        kwargs = {} if lam is None else {"lam": lam}
        _, ret = vector_gae(rt, vt, nv, dt, **kwargs)
        ret = ret.cpu().numpy()
        return {"gae_mc_mae": [float(np.mean(np.abs(ret[:, :, j] - mc[:, :, j]))) for j in range(HEADS)],
                "gae_mc_bias": [float(np.mean(ret[:, :, j] - mc[:, :, j])) for j in range(HEADS)],
                "value_mc_mae": [float(np.mean(np.abs(v[:, :, j] - mc[:, :, j]))) for j in range(HEADS)]}

    def rollout(self, env, obs):
        od = obs.shape[-1]
        out = {}
        for tag, (run, lam) in self.arms.items():
            out[tag] = {}
            for bi, lab in enumerate(ORDER):
                out[tag][lab] = {str(snap): self.arm_snapshot(env, od, run, lam, lab, bi, snap)
                                 for snap in SNAPS}
        self.write(out)
        for s in SNAPS:
            for tag in self.arms:
                a, b = [], []
                for lab in ORDER:
                    a += out[tag][lab][str(s)]["gae_mc_mae"]
                    b += out[tag][lab][str(s)]["value_mc_mae"]
                print(s, tag, "GAE-MC", round(float(np.mean(a)), 4),
                      "V-MC", round(float(np.mean(b)), 4))
        return out
