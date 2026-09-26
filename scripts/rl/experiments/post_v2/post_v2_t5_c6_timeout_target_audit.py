#!/usr/bin/env python3
"""What target does the critic get when an episode times out rather than fails?

The implementation zeroes the bootstrap on termination and truncation alike.
This records, for every episode end, both the target as written and the target
a truncation-aware rule would give, along with the observation the env returns
at the boundary, so the two can be compared without changing anything.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rl.core.isaac_audit import RUNS, IsaacAudit, obs_tensor

SRC = "post_v2_t5_c5_h16-2026-09-23"
PREFERENCE = np.array([.7, .1, .1, .1], np.float32)
GAMMA = .99
MIN_STEPS = 1100
MAX_EVENTS = 64
FIRST_EVENTS = 8


def terms(raw, names):
    return {n: raw[:, i] for i, n in enumerate(names)}


class TimeoutTargetAudit(IsaacAudit):
    """Critic target at episode boundaries, termination versus truncation."""

    run = "post_v2_t5_c6_critic_target_repr-2026-09-23"
    report = "timeout_target_audit.json"
    schema = "t5_c6_timeout_target_audit_v1"
    reset_seed = 700000
    set_usd_path = False       # this audit ran against Isaac's own A1 asset

    def rollout(self, env, obs):
        from talon_rl.t3b_objectives import normalized_objective_vector
        from talon_rl.t4_actor_critic import T4SharedActorCritic

        u = env.unwrapped
        ad = u.action_manager.total_action_dim
        m = T4SharedActorCritic(obs.shape[-1], ad).cuda()
        m.load_state_dict(torch.load(RUNS / SRC / "T_snap_100.pt", map_location="cuda",
                                     weights_only=False)["model"])
        m.eval()
        w = torch.tensor(PREFERENCE, device="cuda").repeat(self.num_envs, 1)
        mgr = u.reward_manager
        maxlen = int(getattr(u, "max_episode_length", -1))
        dt = float(u.step_dt)

        events = []
        for t in range(max(maxlen + 20, MIN_STEPS)):
            with torch.no_grad():
                v = m.value_with_preference(obs, w)
                a = m.act_inference_with_preference(obs, w)
            nxt, _, term, trunc, info = env.step(a)
            nxt = obs_tensor(nxt).cuda()
            raw = mgr._step_reward.detach().cpu().numpy()
            r = normalized_objective_vector(terms(raw, list(mgr.active_terms)),
                                            shape=(self.num_envs,)) * dt
            with torch.no_grad():
                nv = m.value_with_preference(nxt, w)
            mask = (term | trunc).cpu().numpy()
            for i in np.where(mask)[0]:
                # as implemented the bootstrap is zeroed for both cases
                cur_target = r[i].copy()
                alt_target = r[i] + GAMMA * nv[i].detach().cpu().numpy() * (not bool(term[i]))
                events.append({
                    "t": t, "env": int(i), "term": bool(term[i]), "trunc": bool(trunc[i]),
                    "reward": r[i].tolist(),
                    "v_before": v[i].detach().cpu().tolist(),
                    "v_returned_nextobs": nv[i].detach().cpu().tolist(),
                    "current_target_done_zero": cur_target.tolist(),
                    "term_only_target_using_returned_obs": alt_target.tolist(),
                    "returned_obs_l2": float(nxt[i].norm()),
                    "pre_obs_l2": float(obs[i].norm()),
                    "obs_jump_l2": float((nxt[i] - obs[i]).norm()),
                    "info_keys": sorted(list(info.keys())) if isinstance(info, dict) else str(type(info))})
            obs = nxt

        out = {"schema": self.schema, "max_episode_length": maxlen, "step_dt": dt,
               "event_count": len(events),
               "term_count": sum(e["term"] for e in events),
               "trunc_count": sum(e["trunc"] for e in events),
               "events": events[:MAX_EVENTS]}
        self.write(out)
        print(json.dumps({"max_episode_length": maxlen, "event_count": len(events),
                          "term": out["term_count"], "trunc": out["trunc_count"],
                          "first_events": events[:FIRST_EVENTS]}, indent=2))
        return out


if __name__ == "__main__":
    TimeoutTargetAudit.main()
