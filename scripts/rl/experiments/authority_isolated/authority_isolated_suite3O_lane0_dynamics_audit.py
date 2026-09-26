#!/usr/bin/env python3
"""Which part of u50's command keeps the failing suite-3O lane upright?

Rolls u50, u75 and the repaired policy on the one lane that fails, then rolls
u75 again with u50's command substituted one joint at a time, blended by
weight, and restricted to time windows. Each trial keeps a per-step trace of
height, angular velocity and action, so a rescue can be read as a trajectory
rather than only as a survival bit.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rl.core.isaac_audit import RUNS, IsaacAudit, obs_tensor

H1 = "authority_isolated_h1-2026-09-25"
REPAIR = "authority_isolated_projected_tail_descent-2026-09-25/model_10.pt"
SEED = 840004
PREFERENCE = [.1, .1, .7, .1]
LANE = 0
HORIZON = 24
ALPHAS = [.1, .25, .5, .75, 1.0]
WINDOWS = [(0, 5), (4, 8), (6, 10), (9, 13), (11, 15)]


class Suite3OLane0DynamicsAudit(IsaacAudit):
    """Per-joint, per-weight and per-window rescue of the failing lane."""

    run = "authority_isolated_suite3O_lane0_dynamics_audit-2026-09-25"
    report = "dynamics_audit.json"

    def actor(self, path, obs_dim):
        from talon_rl.authority_isolated_actor_critic import AuthorityIsolatedActorCritic

        m = AuthorityIsolatedActorCritic(obs_dim, 12).cuda()
        m.load_state_dict(torch.load(RUNS / path, map_location="cuda",
                                     weights_only=False)["model"])
        m.eval()
        return m

    def wide(self, path, obs_dim):
        from talon_rl.authority_isolated_wide_critic import AuthorityIsolatedWideCritic

        m = AuthorityIsolatedWideCritic(obs_dim, 12).cuda()
        m.load_state_dict(torch.load(RUNS / path, map_location="cuda",
                                     weights_only=False)["model"])
        m.eval()
        return m

    def trial(self, env, base, donor=None, idx=None, alpha=None, window=None):
        """Roll `base`, optionally taking part of `donor`'s command instead.

        `idx` substitutes those coordinates outright, `alpha` blends the whole
        command, and neither means take the donor's command entirely.
        """
        w = torch.tensor(PREFERENCE, device="cuda").repeat(self.num_envs, 1)
        obs, _ = env.reset(seed=SEED)
        obs = obs_tensor(obs).cuda()
        first_fail, trace = None, []
        for t in range(HORIZON):
            d = env.unwrapped.scene["robot"].data
            with torch.no_grad():
                ab = base.act_inference_with_preference(obs, w)
                a = ab.clone()
                if donor is not None and (window is None or window[0] <= t <= window[1]):
                    ad = donor.act_inference_with_preference(obs, w)
                    if idx is None and alpha is None:
                        a = ad
                    elif idx is not None:
                        a[:, idx] = ad[:, idx]
                    else:
                        a = (1 - alpha) * ab + alpha * ad
            trace.append({"t": t, "height": float(d.root_pos_w[LANE, 2]),
                          "ang_vel": d.root_ang_vel_b[LANE].cpu().tolist(),
                          "action": a[LANE].cpu().tolist()})
            nxt, _, te, tr, _ = env.step(a)
            done = (te | tr).cpu().numpy().astype(bool)
            if done[LANE] and first_fail is None:
                first_fail = t
            obs = obs_tensor(nxt).cuda()
        return {"survived": first_fail is None, "first_fail": first_fail, "trace": trace}

    def rollout(self, env, obs):
        od = obs.shape[-1]
        u50 = self.actor(f"{H1}/model_50.pt", od)
        u75 = self.actor(f"{H1}/model_75.pt", od)
        repaired = self.wide(REPAIR, od)

        names = list(env.unwrapped.scene["robot"].data.joint_names)
        res = {"joint_names": names, "tests": {}}

        def record(lab, result):
            res["tests"][lab] = result
            print(lab, result["first_fail"], flush=True)

        for lab, m in (("u50", u50), ("u75", u75), ("repair", repaired)):
            record(lab, self.trial(env, m))
        for j, n in enumerate(names):
            record("u75_plus_u50_" + n, self.trial(env, u75, u50, [j]))
        for a in ALPHAS:
            record(f"u75_u50_interp_{a}", self.trial(env, u75, u50, alpha=a))
        for win in WINDOWS:
            record(f"u50_all_t{win[0]}_{win[1]}",
                   self.trial(env, u75, u50, idx=list(range(12)), window=win))
        self.write(res)
        return res


if __name__ == "__main__":
    Suite3OLane0DynamicsAudit.main()
