#!/usr/bin/env python3
"""Read-only A credit-quality check: does the GAE advantage that drives the actor agree with the realized long-horizon A advantage on the same states?

Same batch as a_credit_audit.py (1024 envs, 50 warm-up steps, 24 scored
steps collected as in train_v4c.py), and the trajectory is then continued
for 255 more steps under the same policy so every scored state has a fixed
256-step Monte Carlo A return. Both advantages use the checkpoint's own
baseline V_A(s_t):
    A_GAE(t)   = the training GAE advantage (24-step window, as trained)
    A_MC256(t) = G_A^MC256(t) - V_A(s_t)   (no bootstrap, stops at true episode end)
Compared only where A is active. Diagnostic only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))  # scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from a_credit_audit import AUDIT_SEED, ITERATIONS, N, WARMUP, ACreditAudit
from g1_evaluate import FOLDS
from rl.core.algorithms.objective_set_ppo import PPOConfig, gae, normalize_advantages, sample_objective_sets
from rl.core.normalization.running import RunningNormalizer

A = 1
HORIZON = 256


def _rank(x):
    r = np.empty(len(x)); r[np.argsort(x)] = np.arange(len(x)); return r


def compare(g, m):
    """g: GAE advantage, m: MC256 advantage, both 1-D over active A samples."""
    agree = np.mean(np.sign(g) == np.sign(m))
    gc, mc = g - g.mean(), m - m.mean()
    return {"n": int(len(g)), "pearson": float(np.corrcoef(g, m)[0, 1]), "spearman": float(np.corrcoef(_rank(g), _rank(m))[0, 1]),
            "sign_agreement_raw": float(agree), "sign_agreement_centered": float(np.mean(np.sign(gc) == np.sign(mc))),
            "gae_pos_mc_neg": float(np.mean((g > 0) & (m < 0))), "gae_neg_mc_pos": float(np.mean((g < 0) & (m > 0))),
            "nmse": float(np.mean((g - m) ** 2) / (np.var(m) + 1e-12)), "scale_ratio_std": float(np.std(g) / (np.std(m) + 1e-12)),
            "mean_gae": float(g.mean()), "mean_mc": float(m.mean())}


class ACreditQuality(ACreditAudit):
    report = "a_credit_quality.json"

    def rows_for(self, env, model, norm):
        from talon_rl.rewards.objectives import NORMALIZATION_DIVISORS, OBJECTIVE_TERMS
        cfg = PPOConfig(); dev = "cuda"; H = cfg.num_steps
        u = env.unwrapped; mgr = u.reward_manager; names = list(mgr.active_terms)
        S = torch.zeros(4, len(names), device=dev)
        for k, terms in enumerate(OBJECTIVE_TERMS.values()):
            for t in terms:
                S[k, names.index(t)] = 1.0
        S /= torch.as_tensor(np.asarray(NORMALIZATION_DIVISORS), device=dev).unsqueeze(-1)
        torch.manual_seed(AUDIT_SEED); gen = torch.Generator().manual_seed(AUDIT_SEED)
        cards = FOLDS[self.fold]["seen"]
        obs, _ = env.reset(seed=AUDIT_SEED)
        ids = torch.arange(4, device=dev).expand(N, -1).contiguous()
        w, mask = sample_objective_sets(N, cards, gen, dev)
        ne = lambda o: torch.as_tensor(norm.transform(o["privileged"].cpu().numpy()), dtype=torch.float32, device=dev)  # noqa: E731
        v_s, r_tr, rA, dd, w_s, m_s = [], [], [], [], [], []
        with torch.no_grad():
            for t in range(WARMUP + H + HORIZON - 1):
                x, e = obs["policy"], ne(obs)
                uu = model._dist(x, e, ids, w).sample()
                v = model.query_values(x, e, ids, w, ids)
                if t == WARMUP + H:
                    last_v = v.clone()
                obs, _, term, trunc, _ = env.step(torch.tanh(uu))
                r = (mgr._step_reward @ S.T) * u.step_dt
                d = term | trunc
                if WARMUP <= t < WARMUP + H:
                    v_s.append(v.clone()); w_s.append(w.clone()); m_s.append(mask.clone())
                    r_tr.append(r + cfg.gamma * v * trunc.unsqueeze(-1).float())
                if t >= WARMUP:
                    rA.append(r[:, A].clone()); dd.append(d.clone())
                if d.any():
                    di = d.nonzero().squeeze(-1); nw, nm = sample_objective_sets(len(di), cards, gen, dev)
                    w = w.clone(); mask = mask.clone(); w[di], mask[di] = nw, nm
        V = torch.stack(v_s); Rtr = torch.stack(r_tr); D = torch.stack(dd); W = torch.stack(w_s); M = torch.stack(m_s)
        _, adv = gae(Rtr, V, last_v, D[:H], cfg.gamma, cfg.lam)
        adv_n = normalize_advantages(adv.flatten(0, 1), W.flatten(0, 1), M.flatten(0, 1)).reshape(adv.shape)
        RA = torch.stack(rA).cpu().numpy(); DD = D.cpu().numpy()
        G = np.zeros((H, N))
        for t in range(H):
            acc = np.zeros(N); alive = np.ones(N, bool)
            for k in range(HORIZON):
                acc += (cfg.gamma ** k) * RA[t + k] * alive
                alive &= ~DD[t + k]
            G[t] = acc
        act = M[:, :, A].cpu().numpy()
        mc = (G - V[:, :, A].cpu().numpy())[act]
        return {"active_A_samples": int(act.sum()),
                "raw_gae_vs_mc": compare(adv[:, :, A].cpu().numpy()[act], mc),
                "normalized_gae_vs_mc": compare(adv_n[:, :, A].cpu().numpy()[act], mc)}

    def rollout(self, env, obs) -> dict:
        from talon_rl.models.authority.teacher_v4 import TeacherV4
        rows = []
        for it in ITERATIONS:
            ck = torch.load(self.run_dir / f"model_{it}.pt", map_location="cuda", weights_only=False)
            model = TeacherV4().cuda(); model.load_state_dict(ck["model"]); model.eval()
            norm = RunningNormalizer(12, center=True); norm.load_state_dict(ck["extrinsics_normalizer"])
            row = {"iteration": it, **self.rows_for(env, model, norm)}
            rows.append(row)
            print("IT", it, json.dumps(row["raw_gae_vs_mc"]), flush=True)
        out = {"schema": "teacher_v4_c_a_credit_quality_v1", "read_only": True, "changes_verdict": False,
               "fold": self.fold, "seed": self.train_seed, "num_envs": N, "warmup": WARMUP, "horizon": HORIZON,
               "audit_seed": AUDIT_SEED, "rows": rows}
        self.write(out)
        return out


if __name__ == "__main__":
    a = ACreditQuality.parse_args((("--fold",), {"choices": tuple(FOLDS), "required": True}),
                                  (("--seed",), {"type": int, "required": True}))
    ACreditQuality(a.fold, a.seed, 300, a.out).execute()
