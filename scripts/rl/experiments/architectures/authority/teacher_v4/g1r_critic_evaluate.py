#!/usr/bin/env python3
"""V4-C G1-R: critic validity of the frozen V4-C checkpoints against a fixed 256-step Monte Carlo target, per docs/contracts/teacher_v4/teacher-v4-c-g1r-critic-contract.md.

Re-runs the G1 critic rollouts (center and heavy endpoints x 4 suites per
set, same seeds) for 64 + 255 steps. Scores V at t = 0..63 against
G_t = sum_{k=0}^{255} g^k r_{t+k}, stopping at a true episode end, never
bootstrapping. Other G1 criteria are taken from the run's g1_evaluation.json
(same deterministic rollouts); the first 64 steps must reproduce the stored
truncated-target EV or the run is marked invalid.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))  # scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from g1_evaluate import FOLDS, G, NENV, ORDER, SUITES, G1Evaluate, center_w, ev, heavy_w, segret
from rl.core.normalization.running import RunningNormalizer

SCORED = 64
HORIZON = 256
ROLL = SCORED + HORIZON - 1


def mc_targets(R, D):
    """R [ROLL,N,K], D [ROLL,N] -> G [SCORED,N,K], terminated-within-window [SCORED,N], effective horizon [SCORED,N]."""
    Gt = np.zeros((SCORED,) + R.shape[1:]); term = np.zeros((SCORED, R.shape[1]), bool); eff = np.zeros((SCORED, R.shape[1]))
    for t in range(SCORED):
        acc = np.zeros(R.shape[1:]); alive = np.ones(R.shape[1], bool); length = np.full(R.shape[1], HORIZON, float)
        for k in range(HORIZON):
            acc += (G ** k) * R[t + k] * alive[:, None]
            ended = alive & D[t + k]
            length[ended] = k + 1
            alive &= ~D[t + k]
        Gt[t] = acc; term[t] = ~alive; eff[t] = length
    return Gt, term, eff


class G1RCritic(G1Evaluate):
    report = "g1r_critic_evaluation.json"

    def evaluate(self, env, ids, w_np, seed):
        ids_t = torch.tensor(ids, device="cuda").repeat(NENV, 1)
        w = torch.tensor(w_np, device="cuda").repeat(NENV, 1)
        obs, _ = env.reset(seed=seed)
        mgr = env.unwrapped.reward_manager
        R, D, V = [], [], []
        with torch.no_grad():
            for t in range(ROLL):
                x, e = obs["policy"], self.e_norm(obs)
                if t < SCORED:
                    V.append(self.model.query_values(x, e, ids_t, w, ids_t).cpu().numpy())
                obs, _, te, tr, _ = env.step(self.act(x, e, ids_t, w))
                raw = mgr._step_reward.detach().cpu().numpy()
                full = self.nov({n: raw[:, i] for i, n in enumerate(mgr.active_terms)}, shape=(NENV,)) * env.unwrapped.step_dt
                R.append(full[:, list(ids)]); D.append((te | tr).cpu().numpy())
        R = np.asarray(R); D = np.asarray(D, bool); V = np.asarray(V)
        Gt, term, eff = mc_targets(R, D)
        m = len(ids)
        H0 = segret(R[:SCORED], D[:SCORED], 0, 32); H1 = segret(R[:SCORED], D[:SCORED], 32, 64)
        return {"V": V, "G": Gt, "term": term, "eff": eff,
                "ev_trunc": [ev(H0[:, :, j], V[:32, :, j]) for j in range(m)] + [ev(H1[:, :, j], V[32:, :, j]) for j in range(m)]}

    def rollout(self, env, obs) -> dict:
        from talon_rl.models.authority.teacher_v4 import TeacherV4
        from talon_rl.rewards.objectives import normalized_objective_vector
        self.nov = normalized_objective_vector
        ck = torch.load(self.ck, map_location="cuda", weights_only=False)
        self.model = TeacherV4().cuda(); self.model.load_state_dict(ck["model"]); self.model.eval()
        self.norm = RunningNormalizer(12, center=True); self.norm.load_state_dict(ck["extrinsics_normalizer"])
        g1 = json.load(open(self.run_dir / "g1_evaluation.json"))
        stored = {tuple(s["ids"]): s for s in g1["sets"]}
        sets = []; si = 0; repro_all = True
        for m in (2, 3, 4):
            role = "heldout" if m == FOLDS[self.fold]["holdout"] else ("seen" if m in FOLDS[self.fold]["seen"] else "other")
            for ids in itertools.combinations(range(4), m):
                labs = [ORDER[i] for i in ids]; rows = []
                for suite in range(SUITES):
                    seed = 860001 + si * 100 + suite
                    rows.append(self.evaluate(env, ids, center_w(m), seed))
                    for h in range(m):
                        rows.append(self.evaluate(env, ids, heavy_w(m, h), seed))
                evs = []; per = {}
                for j, lab in enumerate(labs):
                    for r in rows:
                        for sl in (slice(0, 32), slice(32, 64)):
                            evs.append(ev(r["G"][sl, :, j], r["V"][sl, :, j]))
                    y = np.concatenate([r["G"][:, :, j].reshape(-1) for r in rows]); p = np.concatenate([r["V"][:, :, j].reshape(-1) for r in rows])
                    per[lab] = {"ev_pooled": ev(y, p), "bias": float(np.mean(p - y)), "corr": float(np.corrcoef(p, y)[0, 1]),
                                "var_target": float(np.var(y)), "var_pred": float(np.var(p))}
                ev_trunc = float(np.mean([x for r in rows for x in r["ev_trunc"]]))
                st = stored[tuple(ids)]
                repro = abs(ev_trunc - st["critic"]["ev_mean"]) < 1e-3 * max(1.0, abs(st["critic"]["ev_mean"]))
                repro_all &= repro
                critic = {"ev_mean": float(np.mean(evs)), "negative_fraction": float(np.mean(np.asarray(evs) < 0)),
                          "termination_fraction": float(np.mean([r["term"].mean() for r in rows])),
                          "effective_horizon_mean": float(np.mean([r["eff"].mean() for r in rows]))}
                critic_valid = critic["ev_mean"] > 0 and critic["negative_fraction"] <= .25
                crit = {k: v for k, v in st["criteria"].items() if k != "critic_valid"}
                crit["critic_valid_MC256"] = critic_valid
                sets.append({"ids": list(ids), "labels": labs, "cardinality": m, "role": role, "critic_mc256": critic,
                             "per_objective": per, "reproduces_stored_trunc": repro, "criteria": crit, "pass": bool(all(crit.values()))})
                print("SET", ids, role, json.dumps(critic), "valid", critic_valid, "repro", repro, flush=True)
                si += 1
        held = [x for x in sets if x["role"] == "heldout"]; anchor = next(x for x in sets if x["cardinality"] == 4)
        anchor_pass = anchor["criteria"]["required_semantics"] and anchor["criteria"]["critic_valid_MC256"] and anchor["criteria"]["endpoint_survival"]
        rep = {"schema": "teacher_v4_c_g1r_critic_v1", "contract": "docs/contracts/teacher_v4/teacher-v4-c-g1r-critic-contract.md",
               "fold": self.fold, "seed": self.train_seed, "checkpoint": str(self.ck), "horizon": HORIZON, "gamma": G,
               "determinism_reproduced": bool(repro_all), "sets": sets,
               "summary": {"critic_valid_count": sum(x["criteria"]["critic_valid_MC256"] for x in sets), "set_count": len(sets),
                           "heldout_pass_count": sum(x["pass"] for x in held), "heldout_set_count": len(held),
                           "seen_pass_count": sum(x["pass"] for x in sets if x["role"] == "seen"),
                           "full_set_anchor_pass": bool(anchor_pass),
                           "fold_seed_pass": bool(repro_all and all(x["pass"] for x in held) and anchor_pass)}}
        self.write(rep)
        print("FINAL", rep["summary"], "determinism", repro_all, flush=True)
        return rep


if __name__ == "__main__":
    a = G1RCritic.parse_args((("--fold",), {"choices": tuple(FOLDS), "required": True}),
                             (("--seed",), {"type": int, "required": True}))
    G1RCritic(a.fold, a.seed, 300, a.out).execute()
