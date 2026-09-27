#!/usr/bin/env python3
"""V4-C seed-sensitivity m=4 anchor evaluation at iteration 300 (docs/contracts/teacher_v4/teacher-v4-c-seed-sensitivity-contract.md).

The G1 m=4 endpoint protocol (center and four heavy endpoints, the G1 m=4
suite seeds, 8 envs, 64 steps, the G1 pass rule), G1 authority vs V3 G0, and
MC256 critic validity per objective for A and S on the same rollouts
continued to 319 steps. `--check-stored` compares the endpoint numbers with
the run's stored g1_evaluation.json and fails on any mismatch.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))  # scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from g1_evaluate import G0, NENV, ORDER, PHYS, PROBE, SUITES, G1Evaluate, center_w, ev, heavy_w, tilt_deg
from g1r_critic_evaluate import ROLL, SCORED, mc_targets
from rl.core.normalization.running import RunningNormalizer

IDS = (0, 1, 2, 3)
M4_SET_INDEX = 10


class AnchorEvaluate(G1Evaluate):
    report = "anchor_evaluation.json"

    def run_pref(self, env, w_np, seed):
        ids_t = torch.tensor(IDS, device="cuda").repeat(NENV, 1)
        w = torch.tensor(w_np, device="cuda").repeat(NENV, 1)
        obs, _ = env.reset(seed=seed)
        mgr, robot = env.unwrapped.reward_manager, env.unwrapped.scene["robot"]
        R, D, V, phys = [], [], [], []
        prev = torch.zeros((NENV, env.unwrapped.action_manager.total_action_dim), device="cuda")
        with torch.no_grad():
            for t in range(ROLL):
                x, e = obs["policy"], self.e_norm(obs)
                if t < SCORED:
                    V.append(self.model.query_values(x, e, ids_t, w, ids_t).cpu().numpy())
                a = self.act(x, e, ids_t, w)
                obs, _, te, tr, _ = env.step(a)
                raw = mgr._step_reward.detach().cpu().numpy()
                R.append(self.nov({n: raw[:, i] for i, n in enumerate(mgr.active_terms)}, shape=(NENV,)) * env.unwrapped.step_dt)
                D.append((te | tr).cpu().numpy())
                if t < SCORED:  # same physical metrics as g1_evaluate.evaluate
                    d = robot.data; cmd = env.unwrapped.command_manager.get_command("base_velocity")
                    vx = float((d.root_lin_vel_b[:, 0] - cmd[:, 0]).abs().mean()); wz = float((d.root_ang_vel_b[:, 2] - cmd[:, 2]).abs().mean())
                    phys.append({"tracking_error": vx + wz, "ang_vel_xy": float(torch.linalg.vector_norm(d.root_ang_vel_b[:, :2], dim=-1).mean()),
                                 "tilt_deg": float(tilt_deg(d.root_quat_w).mean()), "action_rate": float(torch.linalg.vector_norm(a - prev, dim=-1).mean())})
                    prev = a
        R = np.asarray(R); D = np.asarray(D, bool); V = np.asarray(V)
        Gt, _, _ = mc_targets(R, D)
        return {"J": R[:SCORED].mean((0, 1)), "phys": {k: float(np.mean([p[k] for p in phys])) for k in phys[0]},
                "survival": float(1 - D[:SCORED].any(0).mean()), "V": V, "G": Gt}

    def rollout(self, env, obs) -> dict:
        from talon_rl.models.authority.objective_set import ObjectiveSetAuthorityIsolatedWideCritic
        from talon_rl.models.authority.teacher_v4 import TeacherV4
        from talon_rl.rewards.objectives import normalized_objective_vector
        self.nov = normalized_objective_vector
        ck = torch.load(self.ck, map_location="cuda", weights_only=False)
        self.model = TeacherV4().cuda(); self.model.load_state_dict(ck["model"]); self.model.eval()
        self.norm = RunningNormalizer(12, center=True); self.norm.load_state_dict(ck["extrinsics_normalizer"])
        robot = env.unwrapped.scene["robot"]
        e_raw = obs["privileged"][0].clone()
        e_raw[8] = robot.data.default_mass[0, robot.find_bodies("trunk")[0][0]].to(e_raw.device)
        self.probe = torch.tensor(np.load(PROBE)["obs"], device="cuda", dtype=torch.float32)
        self.probe_e = torch.as_tensor(self.norm.transform(e_raw.cpu().numpy()[None]), dtype=torch.float32, device="cuda").repeat(len(self.probe), 1)
        self.g0 = ObjectiveSetAuthorityIsolatedWideCritic(48, 12).cuda()
        self.g0.load_state_dict(torch.load(G0, map_location="cuda", weights_only=False)["model"]); self.g0.eval()
        prefs = {"C": center_w(4), **{ORDER[h]: heavy_w(4, h) for h in range(4)}}
        res = {k: [self.run_pref(env, wv, 860001 + M4_SET_INDEX * 100 + s) for s in range(SUITES)] for k, wv in prefs.items()}

        endpoint = {}
        for h, lab in enumerate(ORDER):
            do = [res[lab][s]["J"][h] - res["C"][s]["J"][h] for s in range(SUITES)]
            dp = [res[lab][s]["phys"][PHYS[lab]] - res["C"][s]["phys"][PHYS[lab]] for s in range(SUITES)]
            oo, pp = np.mean(np.asarray(do) > 0), np.mean(np.asarray(dp) < 0)
            ms = min(r["survival"] for r in res[lab])
            endpoint[lab] = {"objective_correct_fraction": float(oo), "physical_correct_fraction": float(pp),
                             "mean_objective_delta": float(np.mean(do)), "mean_physical_delta": float(np.mean(dp)),
                             "min_survival": float(ms), "pass": bool(oo >= .75 and pp >= .75 and ms >= .95)}
        critic = {}
        for h, lab in enumerate(ORDER):
            evs = [ev(r["G"][sl, :, h], r["V"][sl, :, h]) for k in res for r in res[k] for sl in (slice(0, 32), slice(32, 64))]
            critic[lab] = {"ev_mean": float(np.mean(evs)), "negative_fraction": float(np.mean(np.asarray(evs) < 0)),
                           "valid": bool(np.mean(evs) > 0 and np.mean(np.asarray(evs) < 0) <= .25)}
        auth = self.authority(IDS)
        omega = {k: float(np.mean([r["phys"]["ang_vel_xy"] for r in res[k]])) for k in res}
        out = {"schema": "teacher_v4_c_anchor_evaluation_v1", "fold": self.fold, "seed": self.train_seed, "checkpoint": str(self.ck),
               "checkpoint_sha256": self.sha(self.ck), "endpoint": endpoint, "critic_mc256": critic,
               "authority": {k: auth[k] for k in ("pairwise_retention", "tangent_retention")},
               "omega_xy": omega, "dJ_A": endpoint["A"]["mean_objective_delta"],
               "min_endpoint_survival": float(min(e["min_survival"] for e in endpoint.values()))}
        if self.check_stored:
            st = next(s for s in json.load(open(self.run_dir / "g1_evaluation.json"))["sets"] if s["cardinality"] == 4)["endpoint"]
            bad = [(lab, k) for lab in ORDER for k in ("objective_correct_fraction", "physical_correct_fraction", "mean_objective_delta", "mean_physical_delta", "min_survival")
                   if abs(st[lab][k] - endpoint[lab][k]) > 1e-6 * max(1.0, abs(st[lab][k]))]
            out["stored_check_mismatches"] = bad
            if bad:
                self.write(out)
                raise SystemExit(f"endpoint values differ from stored g1_evaluation.json: {bad}")
        self.write(out)
        print("ANCHOR", self.fold, self.train_seed, {k: v["pass"] for k, v in endpoint.items()}, {k: v["valid"] for k, v in critic.items()}, flush=True)
        return out


if __name__ == "__main__":
    a = AnchorEvaluate.parse_args((("--fold",), {"choices": ("G1-2", "G1-3"), "required": True}),
                                  (("--seed",), {"type": int, "required": True}),
                                  (("--check-stored",), {"action": "store_true"}))
    ev_ = AnchorEvaluate(a.fold, a.seed, 300, a.out)
    ev_.check_stored = a.check_stored
    ev_.execute()
