#!/usr/bin/env python3
"""Post-hoc V4-C C1-D0 diagnostic: critic EV against the registered truncated 32-step target versus the bootstrapped segment target, on the same G1 evaluation rollouts.

Does not change the V4-C G1 verdict. Re-runs only the rollouts the G1 critic
gate uses (center and heavy endpoints x 4 suites per set, same seeds, same
deterministic policy) and, per objective, reports for both targets:

    G_trunc(t) = sum_{k=0}^{end-1-t} g^k r_{t+k}
    G_boot(t)  = G_trunc(t) + g^(end-t) V_q(s_end) * [no episode end in t..end-1]

where V_q(s_end) is the query value of the same objective under the same
objective set at the state after the segment's last step. The truncated EV
is also checked against the stored g1_evaluation.json, so the rollouts are
known to be the same ones.
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

from g1_evaluate import FOLDS, G, NENV, ORDER, STEPS, SUITES, G1Evaluate, center_w, heavy_w
from rl.core.normalization.running import RunningNormalizer


def _ev(y, p):
    y = np.asarray(y, float).reshape(-1); p = np.asarray(p, float).reshape(-1)
    return float(1 - np.var(y - p) / (np.var(y) + 1e-12))


def _targets(R, D, V, v_end, st, en):
    """R, V [T,N,K], D [T,N], v_end [N,K] -> truncated and bootstrapped targets [en-st,N,K]."""
    trunc = np.zeros_like(R[st:en]); run = np.zeros_like(R[0]); alive = np.ones(R.shape[1], bool)
    boot = np.zeros_like(trunc); steps_left_disc = np.zeros(R.shape[1])
    for t in range(en - 1, st - 1, -1):
        run = R[t] + G * run * (~D[t])[:, None]
        alive = alive & ~D[t]
        trunc[t - st] = run
        boot[t - st] = run + (G ** (en - t)) * v_end * alive[:, None]
    return trunc, boot


class CriticTargetDiagnostic(G1Evaluate):
    report = "critic_target_diagnostic.json"

    def evaluate(self, env, ids, w_np, seed):
        ids_t = torch.tensor(ids, device="cuda").repeat(NENV, 1)
        w = torch.tensor(w_np, device="cuda").repeat(NENV, 1)
        obs, _ = env.reset(seed=seed)
        mgr = env.unwrapped.reward_manager
        R, D, V = [], [], []
        with torch.no_grad():
            for _ in range(STEPS):
                x, e = obs["policy"], self.e_norm(obs)
                V.append(self.model.query_values(x, e, ids_t, w, ids_t).cpu().numpy())
                obs, _, te, tr, _ = env.step(self.act(x, e, ids_t, w))
                raw = mgr._step_reward.detach().cpu().numpy()
                full = self.nov({n: raw[:, i] for i, n in enumerate(mgr.active_terms)}, shape=(NENV,)) * env.unwrapped.step_dt
                R.append(full[:, list(ids)]); D.append((te | tr).cpu().numpy())
            v64 = self.model.query_values(obs["policy"], self.e_norm(obs), ids_t, w, ids_t).cpu().numpy()
        R = np.asarray(R); D = np.asarray(D, bool); V = np.asarray(V)
        out = {"trunc": [], "boot": [], "pred": []}
        for st, en, v_end in ((0, 32, V[32]), (32, 64, v64)):
            tr_, bo_ = _targets(R, D, V, v_end, st, en)
            out["trunc"].append(tr_); out["boot"].append(bo_); out["pred"].append(V[st:en])
        return out

    def rollout(self, env, obs) -> dict:
        from talon_rl.models.authority.teacher_v4 import TeacherV4
        from talon_rl.rewards.objectives import normalized_objective_vector
        self.nov = normalized_objective_vector
        ck = torch.load(self.ck, map_location="cuda", weights_only=False)
        self.model = TeacherV4().cuda(); self.model.load_state_dict(ck["model"]); self.model.eval()
        self.norm = RunningNormalizer(12, center=True); self.norm.load_state_dict(ck["extrinsics_normalizer"])
        stored = {tuple(s["ids"]): s["critic"] for s in json.load(open(self.run_dir / "g1_evaluation.json"))["sets"]}
        sets = []; si = 0
        for m in (2, 3, 4):
            role = "heldout" if m == FOLDS[self.fold]["holdout"] else ("seen" if m in FOLDS[self.fold]["seen"] else "other")
            for ids in itertools.combinations(range(4), m):
                labs = [ORDER[i] for i in ids]
                # the G1 critic gate pools EV over these exact rollouts, per objective and segment
                rows = []
                for suite in range(SUITES):
                    seed = 860001 + si * 100 + suite
                    rows.append(self.evaluate(env, ids, center_w(m), seed))
                    for h in range(m):
                        rows.append(self.evaluate(env, ids, heavy_w(m, h), seed))
                ev_t, ev_b, per = [], [], {}
                for j, lab in enumerate(labs):
                    y_t = np.concatenate([np.concatenate([r["trunc"][s][:, :, j] for s in (0, 1)]).reshape(-1) for r in rows])
                    y_b = np.concatenate([np.concatenate([r["boot"][s][:, :, j] for s in (0, 1)]).reshape(-1) for r in rows])
                    p = np.concatenate([np.concatenate([r["pred"][s][:, :, j] for s in (0, 1)]).reshape(-1) for r in rows])
                    per[lab] = {"ev_trunc_pooled": _ev(y_t, p), "ev_boot_pooled": _ev(y_b, p),
                                "bias_trunc": float(np.mean(p - y_t)), "bias_boot": float(np.mean(p - y_b)),
                                "corr_trunc": float(np.corrcoef(p, y_t)[0, 1]), "corr_boot": float(np.corrcoef(p, y_b)[0, 1]),
                                "var_trunc_target": float(np.var(y_t)), "var_boot_target": float(np.var(y_b)), "var_pred": float(np.var(p))}
                    # the registered statistic: EV per rollout x segment x objective, then averaged
                    for r in rows:
                        for s in (0, 1):
                            ev_t.append(_ev(r["trunc"][s][:, :, j], r["pred"][s][:, :, j]))
                            ev_b.append(_ev(r["boot"][s][:, :, j], r["pred"][s][:, :, j]))
                row = {"ids": list(ids), "labels": labs, "role": role,
                       "registered_style": {"ev_mean_trunc": float(np.mean(ev_t)), "negative_fraction_trunc": float(np.mean(np.asarray(ev_t) < 0)),
                                            "ev_mean_boot": float(np.mean(ev_b)), "negative_fraction_boot": float(np.mean(np.asarray(ev_b) < 0))},
                       "stored_ev_mean": stored[tuple(ids)]["ev_mean"], "per_objective": per}
                row["reproduces_stored_trunc"] = abs(row["registered_style"]["ev_mean_trunc"] - row["stored_ev_mean"]) < 1e-3 * max(1.0, abs(row["stored_ev_mean"]))
                sets.append(row); si += 1
                print("SET", ids, role, json.dumps(row["registered_style"]), "repro", row["reproduces_stored_trunc"], flush=True)
        rep = {"schema": "teacher_v4_c1_d0_critic_target_diagnostic_v1", "post_hoc": True, "changes_verdict": False,
               "fold": self.fold, "seed": self.train_seed, "checkpoint": str(self.ck), "sets": sets}
        self.write(rep)
        return rep


if __name__ == "__main__":
    a = CriticTargetDiagnostic.parse_args((("--fold",), {"choices": tuple(FOLDS), "required": True}),
                                          (("--seed",), {"type": int, "required": True}))
    CriticTargetDiagnostic(a.fold, a.seed, 300, a.out).execute()
