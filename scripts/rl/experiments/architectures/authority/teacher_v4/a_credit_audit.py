#!/usr/bin/env python3
"""Read-only A-credit audit on V4-C checkpoints: per-objective advantage size, weighted surrogate share, and actor-gradient contribution in a training-like PPO batch.

For each checkpoint: warm up 1024 envs for 50 steps, then collect 24 steps
exactly as train_v4c.py does (fold's cardinalities, per-episode sets,
stochastic actions, the checkpoint's own critic for per-objective GAE and
the training advantage normalization). At the unchanged policy (ratio = 1)
the actor loss splits exactly into per-objective parts
    L_i = -mean(card * w_i * mask_i * A_i * ratio),
and each g_i = grad_actor L_i. Reported: |A_i| statistics, weighted share,
||g_i||, gradient share, cos(g_i, g_mixed) and pairwise cosines. Diagnostic
only; env, sampler and torch seeds are fixed so checkpoints and runs see the
same seeds.
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

from g1_evaluate import FOLDS, ORDER, G1Evaluate
from rl.core.algorithms.objective_set_ppo import PPOConfig, gae, normalize_advantages, sample_objective_sets
from rl.core.normalization.running import RunningNormalizer

N = 1024
WARMUP = 50
ITERATIONS = (50, 100, 150)
AUDIT_SEED = 20260927


class ACreditAudit(G1Evaluate):
    report = "a_credit_audit.json"

    def build_env(self):
        import gymnasium as gym
        import talon_rl.tasks.locomotion.a1_env  # noqa: F401
        from talon_rl.tasks.locomotion.a1_env.v4c_env_cfg import TalonV4CEnvCfg
        cfg = TalonV4CEnvCfg(); cfg.scene.num_envs = N; cfg.seed = AUDIT_SEED
        self.cfg = cfg
        env = gym.make(self.task, cfg=cfg)
        obs, _ = env.reset(seed=AUDIT_SEED)
        return env, obs

    def batch(self, env, model, norm):
        from talon_rl.rewards.objectives import NORMALIZATION_DIVISORS, OBJECTIVE_TERMS
        cfg = PPOConfig(); dev = "cuda"
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
        buf = {k: [] for k in ("obs", "env", "w", "mask", "u", "v", "r", "d")}
        with torch.no_grad():
            for t in range(WARMUP + cfg.num_steps):
                x, e = obs["policy"], ne(obs)
                dist = model._dist(x, e, ids, w); uu = dist.sample()
                v = model.query_values(x, e, ids, w, ids)
                obs, _, term, trunc, _ = env.step(torch.tanh(uu))
                r = (mgr._step_reward @ S.T) * u.step_dt + cfg.gamma * v * trunc.unsqueeze(-1).float()
                d = term | trunc
                if t >= WARMUP:
                    for k, val in (("obs", x), ("env", e), ("w", w), ("mask", mask), ("u", uu), ("v", v), ("r", r), ("d", d)):
                        buf[k].append(val.clone())
                if d.any():
                    di = d.nonzero().squeeze(-1); nw, nm = sample_objective_sets(len(di), cards, gen, dev)
                    w = w.clone(); mask = mask.clone(); w[di], mask[di] = nw, nm
            last_v = model.query_values(obs["policy"], ne(obs), ids, w, ids)
        T = {k: torch.stack(v) for k, v in buf.items()}
        _, adv = gae(T["r"], T["v"], last_v, T["d"], cfg.gamma, cfg.lam)
        f = {k: T[k].flatten(0, 1) for k in ("obs", "env", "w", "mask", "u")}
        f["ids"] = ids.repeat(cfg.num_steps, 1)
        f["adv_raw"] = adv.flatten(0, 1)
        f["adv"] = normalize_advantages(f["adv_raw"], f["w"], f["mask"])
        return f

    def audit(self, model, f):
        params = model.actor_parameters()
        dist = model._dist(f["obs"], f["env"], f["ids"], f["w"])
        logp = (dist.log_prob(f["u"]) - model._log_det_jacobian(f["u"])).sum(-1)
        ratio = torch.exp(logp - logp.detach())  # exactly 1, carries the gradient
        m = f["mask"].float(); card = m.sum(-1)
        grads = {}
        for i in range(4):
            Li = -(card * f["w"][:, i] * m[:, i] * f["adv"][:, i] * ratio).mean()
            g = torch.autograd.grad(Li, params, retain_graph=True, allow_unused=True)
            grads[ORDER[i]] = torch.cat([(x if x is not None else torch.zeros_like(p)).reshape(-1) for x, p in zip(g, params)])
        gm = sum(grads.values())
        cos = lambda a, b: float(a @ b / (a.norm() * b.norm() + 1e-12))  # noqa: E731
        norms = {k: float(v.norm()) for k, v in grads.items()}
        out = {"active_fraction": {ORDER[i]: float(m[:, i].mean()) for i in range(4)}}
        for key in ("adv_raw", "adv"):
            a = f[key]
            out[key] = {ORDER[i]: {"mean_abs_active": float(a[:, i][m[:, i] > 0].abs().mean()), "std_active": float(a[:, i][m[:, i] > 0].std())} for i in range(4)}
        contrib = {ORDER[i]: float((card * f["w"][:, i] * m[:, i] * f["adv"][:, i]).abs().mean()) for i in range(4)}
        out["weighted_contribution"] = {"abs_mean": contrib, "share": {k: v / sum(contrib.values()) for k, v in contrib.items()}}
        out["gradient"] = {"norm": norms, "share": {k: v / sum(norms.values()) for k, v in norms.items()},
                           "cos_with_mixed": {k: cos(v, gm) for k, v in grads.items()},
                           "pairwise_cos": {a + b: cos(grads[a], grads[b]) for a, b in itertools.combinations(ORDER, 2)},
                           "mixed_norm": float(gm.norm())}
        return out

    def rollout(self, env, obs) -> dict:
        from talon_rl.models.authority.teacher_v4 import TeacherV4
        rows = []
        for it in ITERATIONS:
            ck = torch.load(self.run_dir / f"model_{it}.pt", map_location="cuda", weights_only=False)
            model = TeacherV4().cuda(); model.load_state_dict(ck["model"]); model.eval()
            norm = RunningNormalizer(12, center=True); norm.load_state_dict(ck["extrinsics_normalizer"])
            row = {"iteration": it, **self.audit(model, self.batch(env, model, norm))}
            rows.append(row)
            g = row["gradient"]
            print("IT", it, "grad share", {k: round(v, 3) for k, v in g["share"].items()}, "cos mixed", {k: round(v, 2) for k, v in g["cos_with_mixed"].items()},
                  "contrib share", {k: round(v, 3) for k, v in row["weighted_contribution"]["share"].items()}, flush=True)
        out = {"schema": "teacher_v4_c_a_credit_audit_v1", "read_only": True, "changes_verdict": False,
               "fold": self.fold, "seed": self.train_seed, "num_envs": N, "warmup": WARMUP, "audit_seed": AUDIT_SEED, "rows": rows}
        self.write(out)
        return out


if __name__ == "__main__":
    a = ACreditAudit.parse_args((("--fold",), {"choices": tuple(FOLDS), "required": True}),
                                (("--seed",), {"type": int, "required": True}))
    ACreditAudit(a.fold, a.seed, 300, a.out).execute()
