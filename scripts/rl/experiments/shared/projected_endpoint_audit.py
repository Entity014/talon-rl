#!/usr/bin/env python3
"""Endpoint survival of a tail-projected wide critic, per preference and seed.

Three scripts ran this same audit against different checkpoints and wrote it
to different run directories; they now subclass this and set only the three
values that actually differed.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rl.core.isaac_audit import RUNS, IsaacAudit, obs_tensor

PREFS = {"T": [.7, .1, .1, .1], "A": [.1, .7, .1, .1], "O": [.1, .1, .7, .1],
         "S": [.1, .1, .1, .7], "C": [.25] * 4}
SEEDS = (840003, 840004)
TAU_PROBE = RUNS / "authority_isolated_tail_budget_calibration-2026-09-25/tail_budget_probe.json"


def termination_flags(env):
    """Which termination term fired, per lane. Terms without a tensor are skipped."""
    out = {}
    tm = env.unwrapped.termination_manager
    for n in tm.active_terms:
        try:
            out[n] = tm.get_term(n).detach().cpu().numpy().astype(bool)
        except Exception:
            pass
    return out


class ProjectedEndpointAudit(IsaacAudit):
    """Endpoint survival of a tail-projected wide critic."""

    checkpoint: str = ""     # .pt under the run directory
    horizon: int = 64

    def load_model(self, obs_dim):
        import json

        from talon_rl.authority_isolated_wide_critic import AuthorityIsolatedWideCritic

        m = AuthorityIsolatedWideCritic(obs_dim, 12).cuda()
        state = torch.load(self.dir / self.checkpoint, map_location="cuda", weights_only=False)
        m.load_state_dict(state["model"])
        m.eval()
        self.tau = torch.tensor(json.loads(TAU_PROBE.read_text())["tau"], device="cuda")
        return m

    def rollout(self, env, obs):
        n = self.num_envs
        m = self.load_model(obs.shape[-1])
        rows = []
        for si, seed in enumerate(SEEDS, 2):
            for lab, wv in PREFS.items():
                w = torch.tensor(wv, device="cuda").repeat(n, 1)
                cur, _ = env.reset(seed=seed)
                cur = obs_tensor(cur).cuda()
                done = np.zeros(n, bool)
                ft = np.full(n, -1, int)
                why = [[] for _ in range(n)]
                tails, fl = [], []
                for t in range(self.horizon):
                    with torch.no_grad():
                        z = m._actor_mean_with_preference(cur, w)
                        a = torch.tanh(z)
                    tails.append(float((z.abs() > self.tau).float().mean().cpu()))
                    if seed == 840004 and lab == "C" and t in (4, 5, 6, 7):
                        fl.append(float(a[0, 0].cpu()))
                    nxt, _, te, tr, _ = env.step(a)
                    dd = (te | tr).cpu().numpy().astype(bool)
                    flags = termination_flags(env)
                    for i in range(n):
                        if dd[i] and not done[i]:
                            ft[i] = t
                            why[i] = [k for k, v in flags.items() if v[i]]
                    done |= dd
                    cur = obs_tensor(nxt).cuda()
                q = {"suite": si, "preference": lab, "survival": float(1 - done.mean()),
                     "fail_count": int(done.sum()), "fail_t": ft.tolist(), "reason": why,
                     "tail_fraction_mean": float(np.mean(tails)), "flhip_t4_7": fl}
                rows.append(q)
                print(q, flush=True)
        rep = {"rows": rows,
               "min_survival": min(x["survival"] for x in rows),
               "failed_lanes": sum(x["fail_count"] for x in rows),
               "mean_tail_fraction": float(np.mean([x["tail_fraction_mean"] for x in rows])),
               "residual_flhip_t4_7": [x["flhip_t4_7"] for x in rows
                                       if x["suite"] == 3 and x["preference"] == "C"][0]}
        self.write(rep)
        import json
        print("SUMMARY", json.dumps(rep, indent=2), flush=True)
        return rep
