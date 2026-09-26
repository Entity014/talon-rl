#!/usr/bin/env python3
"""How much of the u50 hip command does u75 need borrowed back to survive?

On the one failing suite-3 orientation lane, replaces the u75 policy's hip
coordinate with a blend toward u50's, by blend weight and by time window. The
smallest blend and the narrowest window that still survives bound where the
failure lives.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rl.core.isaac_audit import RUNS, IsaacAudit, obs_tensor

RUN = "authority_isolated_h1-2026-09-25"
SEED = 840004
PREFERENCE = [.1, .1, .7, .1]
LANE = 0
HORIZON = 24
JOINTS = [(0, "FL_hip"), (3, "RR_hip")]
ALPHAS = [.1, .25, .5, .75, 1.0]
WINDOWS = [(0, 5), (4, 8), (6, 10), (9, 13), (11, 15)]


class Suite3OHipMarginAudit(IsaacAudit):
    """Hip-coordinate margin between u50 and u75 on the failing suite-3 lane."""

    run = "authority_isolated_suite3O_hip_margin_audit-2026-09-25"
    report = "hip_margin.json"

    def policy(self, name, obs_dim):
        from talon_rl.authority_isolated_actor_critic import AuthorityIsolatedActorCritic

        m = AuthorityIsolatedActorCritic(obs_dim, 12).cuda()
        m.load_state_dict(torch.load(RUNS / RUN / name, map_location="cuda",
                                     weights_only=False)["model"])
        m.eval()
        return m

    def trial(self, env, base, donor, j, alpha=1.0, window=None):
        """Roll `base`, replacing coordinate j with a blend toward `donor`."""
        w = torch.tensor(PREFERENCE, device="cuda").repeat(self.num_envs, 1)
        obs, _ = env.reset(seed=SEED)
        obs = obs_tensor(obs).cuda()
        first_fail = None
        for t in range(HORIZON):
            with torch.no_grad():
                a = base.act_inference_with_preference(obs, w)
                q = donor.act_inference_with_preference(obs, w)
                if window is None or window[0] <= t <= window[1]:
                    a[:, j] = (1 - alpha) * a[:, j] + alpha * q[:, j]
            nxt, _, te, tr, _ = env.step(a)
            done = (te | tr).cpu().numpy().astype(bool)
            if done[LANE] and first_fail is None:
                first_fail = t
            obs = obs_tensor(nxt).cuda()
        return {"survived": first_fail is None, "first_fail": first_fail}

    def rollout(self, env, obs):
        base = self.policy("model_75.pt", obs.shape[-1])
        donor = self.policy("model_50.pt", obs.shape[-1])
        rows = []
        for j, name in JOINTS:
            for alpha in ALPHAS:
                q = self.trial(env, base, donor, j, alpha, None)
                q.update(joint=name, alpha=alpha, window="all")
                rows.append(q)
                print(q, flush=True)
            for win in WINDOWS:
                q = self.trial(env, base, donor, j, 1.0, win)
                q.update(joint=name, alpha=1.0, window=f"{win[0]}-{win[1]}")
                rows.append(q)
                print(q, flush=True)
        rep = {"rows": rows}
        self.write(rep)
        return rep


if __name__ == "__main__":
    Suite3OHipMarginAudit.main()
