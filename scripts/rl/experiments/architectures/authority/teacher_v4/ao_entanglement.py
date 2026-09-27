#!/usr/bin/env python3
"""Read-only A-O entanglement diagnostic on a frozen V4-C checkpoint (m=4 set): reward overlap, behavioral separation, outcome separation, each ranked against the other objective pairs.

Does not change any verdict. Uses the G1 m=4 rollouts (center and the four
heavy endpoints, the G1 m=4 suite seeds, 8 envs x 64 steps) and the G1
probe states with the physical-nominal e_t. The V3 G0 (Phase-1) model's
probe action distances are reported as a reference for whether the same
pair structure exists before V4. No thresholds: every A-O number is
reported with its rank among the six objective pairs.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))  # scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from g1_evaluate import G0, NENV, ORDER, PROBE, STEPS, SUITES, G1Evaluate, center_w, heavy_w, tilt_deg
from rl.core.normalization.running import RunningNormalizer

IDS = (0, 1, 2, 3)
PAIRS = list(itertools.combinations(range(4), 2))
M4_SET_INDEX = 10  # position of (0,1,2,3) in the G1 set order, so the suite seeds match G1


def _rank(vals: dict, key) -> dict:
    order = sorted(vals, key=vals.get)
    return {"value": vals[key], "rank_ascending": order.index(key) + 1, "of": len(vals)}


def _pair_name(p): return ORDER[p[0]] + ORDER[p[1]]


class AOEntanglement(G1Evaluate):
    report = "ao_entanglement.json"

    def rollout_full(self, env, w_np, seed):
        ids_t = torch.tensor(IDS, device="cuda").repeat(NENV, 1)
        w = torch.tensor(w_np, device="cuda").repeat(NENV, 1)
        obs, _ = env.reset(seed=seed)
        mgr, robot = env.unwrapped.reward_manager, env.unwrapped.scene["robot"]
        R, P, X = [], [], []
        with torch.no_grad():
            for _ in range(STEPS):
                x, e = obs["policy"], self.e_norm(obs)
                X.append((x.clone(), e.clone()))
                obs, *_ = env.step(self.act(x, e, ids_t, w))
                raw = mgr._step_reward.detach().cpu().numpy()
                R.append(self.nov({n: raw[:, i] for i, n in enumerate(mgr.active_terms)}, shape=(NENV,)) * env.unwrapped.step_dt)
                d = robot.data
                P.append(np.stack([torch.linalg.vector_norm(d.root_ang_vel_b[:, :2], dim=-1).cpu().numpy(),
                                   tilt_deg(d.root_quat_w).cpu().numpy()], -1))
        return np.asarray(R), np.asarray(P), X

    def rollout(self, env, obs) -> dict:
        from talon_rl.models.authority.objective_set import ObjectiveSetAuthorityIsolatedWideCritic, canonical_tokens
        from talon_rl.models.authority.teacher_v4 import TeacherV4
        from talon_rl.rewards.objectives import normalized_objective_vector
        self.nov = normalized_objective_vector
        ck = torch.load(self.ck, map_location="cuda", weights_only=False)
        self.model = TeacherV4().cuda(); self.model.load_state_dict(ck["model"]); self.model.eval()
        self.norm = RunningNormalizer(12, center=True); self.norm.load_state_dict(ck["extrinsics_normalizer"])
        robot = env.unwrapped.scene["robot"]
        e_raw = obs["privileged"][0].clone()
        e_raw[8] = robot.data.default_mass[0, robot.find_bodies("trunk")[0][0]].to(e_raw.device)
        e_ref = torch.as_tensor(self.norm.transform(e_raw.cpu().numpy()[None]), dtype=torch.float32, device="cuda")

        prefs = {"C": center_w(4), **{ORDER[h]: heavy_w(4, h) for h in range(4)}}
        runs = {k: [] for k in prefs}
        for suite in range(SUITES):
            seed = 860001 + M4_SET_INDEX * 100 + suite
            for k, wv in prefs.items():
                runs[k].append(self.rollout_full(env, wv, seed))

        # 1. reward overlap: per-step objective correlation, pooled over all rollouts
        Rall = np.concatenate([r[0].reshape(-1, 4) for k in runs for r in runs[k]])
        corr = np.corrcoef(Rall.T)
        rc = {_pair_name(p): float(corr[p]) for p in PAIRS}
        per_suite = [{_pair_name(p): float(np.corrcoef(np.concatenate([runs[k][s][0].reshape(-1, 4) for k in runs]).T)[p]) for p in PAIRS}
                     for s in range(SUITES)]

        # 2. behavioral separation: action distance between heavy endpoints at matched states
        probe = torch.tensor(np.load(PROBE)["obs"], device="cuda", dtype=torch.float32)
        ids_p = torch.tensor(IDS, device="cuda").repeat(len(probe), 1)
        ep = e_ref.repeat(len(probe), 1)
        center_states = [(x, e) for r in runs["C"] for (x, e) in r[2]]
        xs = torch.cat([x for x, _ in center_states]); es = torch.cat([e for _, e in center_states])
        ids_s = torch.tensor(IDS, device="cuda").repeat(len(xs), 1)
        base = canonical_tokens(device="cuda")[torch.tensor(IDS, device="cuda")]
        g0 = ObjectiveSetAuthorityIsolatedWideCritic(48, 12).cuda()
        g0.load_state_dict(torch.load(G0, map_location="cuda", weights_only=False)["model"]); g0.eval()
        act = {"probe": {}, "rollout": {}, "g0_probe": {}}
        with torch.no_grad():
            for h in range(4):
                wp = torch.tensor(heavy_w(4, h), device="cuda")
                act["probe"][h] = self.model.act_inference(probe, ep, ids_p, wp.repeat(len(probe), 1))
                act["rollout"][h] = self.model.act_inference(xs, es, ids_s, wp.repeat(len(xs), 1))
                act["g0_probe"][h] = g0.act_inference_from_set(probe, base.unsqueeze(0).repeat(len(probe), 1, 1), wp.repeat(len(probe), 1))
        dist = {src: {_pair_name(p): float(torch.linalg.vector_norm(a[p[0]] - a[p[1]], dim=-1).mean()) for p in PAIRS} for src, a in act.items()}

        # 3. outcome separation: distance between heavy endpoints' mean objective vectors, per suite
        outcome = {}
        for p in PAIRS:
            a, b = ORDER[p[0]], ORDER[p[1]]
            outcome[_pair_name(p)] = float(np.mean([np.linalg.norm(runs[a][s][0].mean((0, 1)) - runs[b][s][0].mean((0, 1))) for s in range(SUITES)]))
        phys = {k: {"ang_vel_xy": float(np.mean([r[1][..., 0].mean() for r in runs[k]])), "tilt_deg": float(np.mean([r[1][..., 1].mean() for r in runs[k]])),
                    "objective_mean": np.mean([r[0].mean((0, 1)) for r in runs[k]], 0).tolist()} for k in runs}

        out = {"schema": "teacher_v4_c_ao_entanglement_v1", "read_only": True, "changes_verdict": False,
               "fold": self.fold, "seed": self.train_seed, "checkpoint": str(self.ck),
               "reward_corr": rc, "reward_corr_per_suite": per_suite,
               "action_distance": dist, "outcome_distance": outcome, "per_endpoint": phys,
               "AO": {"reward_corr_rank_by_abs": _rank({k: abs(v) for k, v in rc.items()}, "AO"),
                      "action_distance_probe": _rank(dist["probe"], "AO"),
                      "action_distance_rollout": _rank(dist["rollout"], "AO"),
                      "action_distance_g0_probe": _rank(dist["g0_probe"], "AO"),
                      "outcome_distance": _rank(outcome, "AO")}}
        self.write(out)
        print("AO", out["AO"], flush=True)
        return out


if __name__ == "__main__":
    a = AOEntanglement.parse_args((("--fold",), {"choices": ("G1-2", "G1-3"), "required": True}),
                                  (("--seed",), {"type": int, "required": True}))
    AOEntanglement(a.fold, a.seed, 300, a.out).execute()
