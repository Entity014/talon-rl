#!/usr/bin/env python3
"""Read-only A timeline on one V4-C run: at each saved checkpoint, the m=4 A endpoint (A-heavy vs center), A authority on the probes, survival, and the A critic's MC256 validity.

Diagnostic only (checkpoints 50..250 are never selectable under the G1
contract). Per checkpoint, with the G1 m=4 suite seeds and the physical
nominal probe e_t:
- dJ_A: A objective, A-heavy minus center, over the G1 64-step window;
- d_omega_improvement: center minus A-heavy mean ||omega_xy|| (positive = A-heavy better);
- survival of A-heavy and center over 64 steps;
- authority: mean ||a(A-heavy) - a(center)|| on the probe states;
- critic A: EV of V_A against the fixed 256-step MC target (G1-R form) over
  the center and four heavy rollouts, registered pooling and pooled.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))  # scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from g1_evaluate import NENV, ORDER, PROBE, SUITES, G1Evaluate, center_w, ev, heavy_w
from g1r_critic_evaluate import ROLL, SCORED, mc_targets
from rl.core.normalization.running import RunningNormalizer

IDS = (0, 1, 2, 3)
A = 1
M4_SET_INDEX = 10
ITERATIONS = (50, 100, 150, 200, 250, 300)


class ACheckpointLadder(G1Evaluate):
    report = "a_checkpoint_ladder.json"

    def run_pref(self, env, w_np, seed):
        ids_t = torch.tensor(IDS, device="cuda").repeat(NENV, 1)
        w = torch.tensor(w_np, device="cuda").repeat(NENV, 1)
        obs, _ = env.reset(seed=seed)
        mgr, robot = env.unwrapped.reward_manager, env.unwrapped.scene["robot"]
        R, D, V, om = [], [], [], []
        with torch.no_grad():
            for t in range(ROLL):
                x, e = obs["policy"], self.e_norm(obs)
                if t < SCORED:
                    V.append(self.model.query_values(x, e, ids_t, w, ids_t).cpu().numpy())
                obs, _, te, tr, _ = env.step(self.act(x, e, ids_t, w))
                raw = mgr._step_reward.detach().cpu().numpy()
                R.append(self.nov({n: raw[:, i] for i, n in enumerate(mgr.active_terms)}, shape=(NENV,)) * env.unwrapped.step_dt)
                D.append((te | tr).cpu().numpy())
                if t < SCORED:
                    om.append(torch.linalg.vector_norm(robot.data.root_ang_vel_b[:, :2], dim=-1).cpu().numpy())
        R = np.asarray(R); D = np.asarray(D, bool); V = np.asarray(V)
        Gt, _, _ = mc_targets(R, D)
        return {"J": R[:SCORED].mean((0, 1)), "omega": float(np.mean(om)), "survival": float(1 - D[:SCORED].any(0).mean()),
                "V": V, "G": Gt}

    def rollout(self, env, obs) -> dict:
        from talon_rl.models.authority.teacher_v4 import TeacherV4
        from talon_rl.rewards.objectives import normalized_objective_vector
        self.nov = normalized_objective_vector
        robot = env.unwrapped.scene["robot"]
        e_raw = obs["privileged"][0].clone()
        e_raw[8] = robot.data.default_mass[0, robot.find_bodies("trunk")[0][0]].to(e_raw.device)
        probe = torch.tensor(np.load(PROBE)["obs"], device="cuda", dtype=torch.float32)
        ids_p = torch.tensor(IDS, device="cuda").repeat(len(probe), 1)
        prefs = {"C": center_w(4), **{ORDER[h]: heavy_w(4, h) for h in range(4)}}
        rows = []
        for it in ITERATIONS:
            ck = torch.load(self.run_dir / f"model_{it}.pt", map_location="cuda", weights_only=False)
            self.model = TeacherV4().cuda(); self.model.load_state_dict(ck["model"]); self.model.eval()
            self.norm = RunningNormalizer(12, center=True); self.norm.load_state_dict(ck["extrinsics_normalizer"])
            ep = torch.as_tensor(self.norm.transform(e_raw.cpu().numpy()[None]), dtype=torch.float32, device="cuda").repeat(len(probe), 1)
            res = {k: [self.run_pref(env, wv, 860001 + M4_SET_INDEX * 100 + s) for s in range(SUITES)] for k, wv in prefs.items()}
            dJ = [float(res["A"][s]["J"][A] - res["C"][s]["J"][A]) for s in range(SUITES)]
            dW = [res["C"][s]["omega"] - res["A"][s]["omega"] for s in range(SUITES)]
            with torch.no_grad():
                aA = self.model.act_inference(probe, ep, ids_p, torch.tensor(heavy_w(4, A), device="cuda").repeat(len(probe), 1))
                aC = self.model.act_inference(probe, ep, ids_p, torch.tensor(center_w(4), device="cuda").repeat(len(probe), 1))
            evs = [ev(r["G"][sl, :, A], r["V"][sl, :, A]) for k in res for r in res[k] for sl in (slice(0, 32), slice(32, 64))]
            y = np.concatenate([r["G"][:, :, A].reshape(-1) for k in res for r in res[k]])
            p = np.concatenate([r["V"][:, :, A].reshape(-1) for k in res for r in res[k]])
            row = {"iteration": it,
                   "dJ_A_mean": float(np.mean(dJ)), "dJ_A_correct_fraction": float(np.mean(np.asarray(dJ) > 0)),
                   "d_omega_improvement_mean": float(np.mean(dW)), "d_omega_correct_fraction": float(np.mean(np.asarray(dW) > 0)),
                   "omega_center": float(np.mean([r["omega"] for r in res["C"]])), "omega_A_heavy": float(np.mean([r["omega"] for r in res["A"]])),
                   "omega_O_heavy": float(np.mean([r["omega"] for r in res["O"]])),
                   "survival_A_heavy": float(min(r["survival"] for r in res["A"])), "survival_center": float(min(r["survival"] for r in res["C"])),
                   "authority_A_vs_center": float(torch.linalg.vector_norm(aA - aC, dim=-1).mean()),
                   "critic_A_ev_mean": float(np.mean(evs)), "critic_A_negative_fraction": float(np.mean(np.asarray(evs) < 0)),
                   "critic_A_ev_pooled": ev(y, p), "critic_A_pred_var_over_target_var": float(np.var(p) / (np.var(y) + 1e-12))}
            rows.append(row)
            print("IT", json.dumps(row), flush=True)
        out = {"schema": "teacher_v4_c_a_checkpoint_ladder_v1", "read_only": True, "changes_verdict": False,
               "fold": self.fold, "seed": self.train_seed, "rows": rows}
        self.write(out)
        return out


if __name__ == "__main__":
    a = ACheckpointLadder.parse_args((("--fold",), {"choices": ("G1-2", "G1-3"), "required": True}),
                                     (("--seed",), {"type": int, "required": True}))
    ACheckpointLadder(a.fold, a.seed, 300, a.out).execute()
