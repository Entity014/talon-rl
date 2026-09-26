#!/usr/bin/env python3
"""Fresh-rollout value replay: how well each snapshot's critic predicts return.

Five scripts ran this same audit over different pairs of training arms. They
subclass this and set only the arms, the schema, the reset seed base and which
arm's run directory the report belongs to.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rl.core.isaac_audit import RUNS, IsaacAudit, obs_tensor

ORDER = ("T", "A", "O", "S")
SNAPS = tuple(range(26))
GAMMA = .99
PREFS = {"T": np.array([.7, .1, .1, .1], np.float32),
         "A": np.array([.1, .7, .1, .1], np.float32),
         "O": np.array([.1, .1, .7, .1], np.float32),
         "S": np.array([.1, .1, .1, .7], np.float32)}
REPORT_SNAPSHOTS = (0, 1, 2, 3, 5, 10, 12, 15, 20, 25)


def terms(raw, names):
    return {n: raw[:, i] for i, n in enumerate(names)}


def ev(y, p):
    """Explained variance of a prediction against its target."""
    y = np.asarray(y, float).reshape(-1)
    p = np.asarray(p, float).reshape(-1)
    return float(1 - np.var(y - p) / (np.var(y) + 1e-12))


def discounted_return(r, done, seg=None):
    """Discounted return per step; `seg` bounds it to fixed-length windows."""
    out = np.zeros_like(r)
    if seg is None:
        run = np.zeros_like(r[0])
        for t in range(len(r) - 1, -1, -1):
            run = r[t] + GAMMA * run * (~done[t])[:, None]
            out[t] = run
    else:
        for st in range(0, len(r), seg):
            en = min(st + seg, len(r))
            run = np.zeros_like(r[0])
            for t in range(en - 1, st - 1, -1):
                run = r[t] + GAMMA * run * (~done[t])[:, None]
                out[t] = run
    return out


class ReplayAudit(IsaacAudit):
    """Fresh-rollout value replay across snapshots, for a pair of arms."""

    arms: dict = {}          # arm name -> run directory under runs/
    schema: str = ""
    seed_base: int = 0
    report = "replay_audit.json"

    def rollout(self, env, obs):
        from talon_rl.t3b_objectives import normalized_objective_vector
        from talon_rl.t4_actor_critic import T4SharedActorCritic

        od = obs.shape[-1]
        ad = env.unwrapped.action_manager.total_action_dim
        mgr = env.unwrapped.reward_manager
        out = {"schema": self.schema, "arms": {}}
        for arm, run in self.arms.items():
            armout = {}
            for bi, lab in enumerate(ORDER):
                w = torch.tensor(PREFS[lab], device="cuda").repeat(8, 1)
                rows = []
                for snap in SNAPS:
                    m = T4SharedActorCritic(od, ad).cuda()
                    m.load_state_dict(torch.load(RUNS / run / f"{lab}_snap_{snap}.pt",
                                                 map_location="cuda", weights_only=False)["model"])
                    m.eval()
                    cur, _ = env.reset(seed=self.seed_base + bi * 1000)
                    cur = obs_tensor(cur).cuda()
                    r, d, v = [], [], []
                    with torch.no_grad():
                        for _ in range(64):
                            v.append(m.value_with_preference(cur, w).cpu().numpy())
                            a = m.act_inference_with_preference(cur, w)
                            nxt, _, te, tr, _ = env.step(a)
                            raw = mgr._step_reward.detach().cpu().numpy()
                            r.append(normalized_objective_vector(terms(raw, list(mgr.active_terms)),
                                                                 shape=(8,)) * env.unwrapped.step_dt)
                            d.append((te | tr).cpu().numpy())
                            cur = obs_tensor(nxt).cuda()
                    r = np.asarray(r)
                    d = np.asarray(d, bool)
                    v = np.asarray(v)
                    h32 = discounted_return(r, d, 32)
                    mc = discounted_return(r, d, None)
                    rows.append({
                        "snapshot": snap,
                        "h32_ev": [ev(h32[:, :, j], v[:, :, j]) for j in range(4)],
                        "mc64_ev": [ev(mc[:, :, j], v[:, :, j]) for j in range(4)],
                        "h32_bias": [float(np.mean(v[:, :, j] - h32[:, :, j])) for j in range(4)],
                        "mc64_bias": [float(np.mean(v[:, :, j] - mc[:, :, j])) for j in range(4)],
                        "survival": float(1 - d.any(0).mean())})
                armout[lab] = rows
            out["arms"][arm] = armout
        out["aggregate"] = self.aggregate(out)
        self.write(out)
        self.report_table(out["aggregate"])
        return out

    def aggregate(self, out):
        agg = {}
        for arm in self.arms:
            traj = []
            for snap in SNAPS:
                h, m, hb, mb, sv = [], [], [], [], []
                by = [[] for _ in range(4)]
                for lab in ORDER:
                    r = out["arms"][arm][lab][snap]
                    h += r["h32_ev"]
                    m += r["mc64_ev"]
                    hb += r["h32_bias"]
                    mb += r["mc64_bias"]
                    sv.append(r["survival"])
                    for j, x in enumerate(r["h32_ev"]):
                        by[j].append(x)
                traj.append({"snapshot": snap,
                             "h32_ev_mean": float(np.mean(h)),
                             "h32_negative_fraction": float(np.mean(np.array(h) < 0)),
                             "h32_ev_by_head": [float(np.mean(x)) for x in by],
                             "mc64_ev_mean": float(np.mean(m)),
                             "mc64_negative_fraction": float(np.mean(np.array(m) < 0)),
                             "h32_mean_abs_bias": float(np.mean(np.abs(hb))),
                             "mc64_mean_abs_bias": float(np.mean(np.abs(mb))),
                             "min_survival": float(np.min(sv))})
            agg[arm] = traj
        return agg

    def report_table(self, agg):
        for arm in self.arms:
            print("\\n", arm)
            for s in REPORT_SNAPSHOTS:
                r = agg[arm][s]
                print(s, "H32", round(r["h32_ev_mean"], 3),
                      "neg", round(r["h32_negative_fraction"], 2),
                      "O", round(r["h32_ev_by_head"][2], 3),
                      "MC", round(r["mc64_ev_mean"], 3),
                      "bias", round(r["h32_mean_abs_bias"], 3),
                      "surv", round(r["min_survival"], 3))
