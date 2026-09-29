#!/usr/bin/env python3
"""FC-A read-only R credit / gradient / virtual-step audit on FB-2a checkpoints
(docs/contracts/teacher_v4/teacher-v4-fca-r-credit-audit-contract.md).

Per checkpoint and fixed preference condition (R vertex, R+, C, O+): a
training-like batch (stochastic policy, 50 warm-up + 24 steps, FB-2a
streams [T_lin, R, O], loss weights [lambda_region, w_R, w_O], the
checkpoint's own critic, GAE, the training advantage normalization).
Layer 1: advantage size and weighted contribution per stream. Layer 2: the
per-stream actor gradients g_T, g_R, g_O at ratio = 1, norms, ratios, and
cosines. Both over all samples and over locomoting envs (rollout-mean
track_lin >= 0.40). Layer 3 (R+ and C): in-memory virtual steps along the
R-objective ascent direction (+/-) and the mixed direction, sized to batch
KL {0.005, 0.01, 0.02}, then a deterministic replay (reset seeds 910001/2,
steady window 33-128): F_rate, F_osc, F_O, tl. Nothing is saved or trained.
"""
from __future__ import annotations

import copy
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))  # scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from g1_evaluate import G1Evaluate
from rl.core.algorithms.objective_set_ppo import PPOConfig, gae, gaussian_kl, normalize_advantages
from rl.core.normalization.running import RunningNormalizer

N = 1024
WARMUP = 50
AUDIT_SEED = 20260929
CONDS = {"R-vertex": (1.0, 0.0), "R+": (0.7, 0.3), "C": (0.5, 0.5), "O+": (0.3, 0.7)}
STEP_CONDS = ("R+", "C")
KLS = (0.005, 0.01, 0.02)
RESET_SEEDS = (910001, 910002)
EDGES = (0.15, 0.40, 0.60, 0.85)  # train_v4c LAGR_EDGES
STREAMS = ("T", "R", "O")
DT, TAU = 0.02, 0.5


def region(wr):
    return int(np.searchsorted(EDGES, wr, side="right"))


class FCACreditAudit(G1Evaluate):
    num_envs = N
    report = "fca_credit_audit.json"

    def build_env(self):
        import gymnasium as gym
        import talon_rl.tasks.locomotion.a1_env  # noqa: F401
        from talon_rl.tasks.locomotion.a1_env.v4c_env_cfg import TalonV4CEnvCfg
        cfg = TalonV4CEnvCfg(); cfg.scene.num_envs = N; cfg.seed = AUDIT_SEED
        cfg.observations.policy.enable_corruption = False
        env = gym.make(self.task, cfg=cfg)
        obs, _ = env.reset(seed=AUDIT_SEED)
        return env, obs

    def ne(self, o):
        return torch.as_tensor(self.norm.transform(o["privileged"].cpu().numpy()), dtype=torch.float32, device="cuda")

    def batch(self, env, model, wv, lam):
        from talon_rl.rewards.objectives import NORMALIZATION_DIVISORS
        cfg = PPOConfig(); dev = "cuda"; u = env.unwrapped; mgr = u.reward_manager; names = list(mgr.active_terms)
        S = torch.zeros(3, len(names), device=dev)
        for k, t in enumerate(("track_lin_vel_xy_exp", "ang_vel_xy_l2", "flat_orientation_l2")):
            S[k, names.index(t)] = 1.0 / float(NORMALIZATION_DIVISORS[[0, 1, 2][k]])
        torch.manual_seed(AUDIT_SEED)
        obs, _ = env.reset(seed=AUDIT_SEED)
        ids = torch.tensor([1, 2], device=dev).repeat(N, 1); qids = torch.arange(3, device=dev).repeat(N, 1)
        w = torch.tensor(wv, device=dev).repeat(N, 1)
        buf = {k: [] for k in ("obs", "env", "u", "v", "r", "d", "tl")}
        with torch.no_grad():
            for t in range(WARMUP + cfg.num_steps):
                x, e = obs["policy"], self.ne(obs)
                dist = model._dist(x, e, ids, w); uu = dist.sample()
                v = model.query_values(x, e, ids, w, qids)
                obs, _, term, trunc, _ = env.step(torch.tanh(uu))
                r = (mgr._step_reward @ S.T) * u.step_dt + cfg.gamma * v * trunc.unsqueeze(-1).float()
                if t >= WARMUP:
                    for k, val in (("obs", x), ("env", e), ("u", uu), ("v", v), ("r", r), ("d", term | trunc),
                                   ("tl", mgr._step_reward[:, names.index("track_lin_vel_xy_exp")])):
                        buf[k].append(val.clone())
            last_v = model.query_values(obs["policy"], self.ne(obs), ids, w, qids)
        T = {k: torch.stack(v) for k, v in buf.items()}
        _, adv = gae(T["r"], T["v"], last_v, T["d"], cfg.gamma, cfg.lam)
        H = cfg.num_steps
        f = {k: T[k].flatten(0, 1) for k in ("obs", "env", "u")}
        f["ids"], f["w"] = ids.repeat(H, 1), w.repeat(H, 1)
        lw = torch.tensor([lam, wv[0], wv[1]], device=dev).repeat(N * H, 1); lm = torch.ones_like(lw, dtype=torch.bool)
        f["lw"] = lw; f["adv_raw"] = adv.flatten(0, 1); f["adv"] = normalize_advantages(f["adv_raw"], lw, lm)
        f["loco"] = (T["tl"].mean(0) >= 0.40).repeat(H)  # env-level rollout mean, per sample
        return f

    def grads(self, model, f, sel):
        params = model.actor_parameters()
        dist = model._dist(f["obs"][sel], f["env"][sel], f["ids"][sel], f["w"][sel])
        logp = (dist.log_prob(f["u"][sel]) - model._log_det_jacobian(f["u"][sel])).sum(-1)
        ratio = torch.exp(logp - logp.detach())
        g = {}
        for i, s in enumerate(STREAMS):
            Li = -(3.0 * f["lw"][sel][:, i] * f["adv"][sel][:, i] * ratio).mean()
            gi = torch.autograd.grad(Li, params, retain_graph=True, allow_unused=True)
            g[s] = torch.cat([(x if x is not None else torch.zeros_like(p)).reshape(-1) for x, p in zip(gi, params)])
        return g

    def layers(self, model, f, sel):
        cos = lambda a, b: float(a @ b / (a.norm() * b.norm() + 1e-12))  # noqa: E731
        g = self.grads(model, f, sel); gm = sum(g.values())
        contrib = {s: float((f["lw"][sel][:, i] * f["adv"][sel][:, i]).abs().mean()) for i, s in enumerate(STREAMS)}
        return {"n": int(sel.sum()),
                "adv_raw_std": {s: float(f["adv_raw"][sel][:, i].std()) for i, s in enumerate(STREAMS)},
                "contribution": contrib, "contribution_share": {s: v / sum(contrib.values()) for s, v in contrib.items()},
                "grad_norm": {s: float(v.norm()) for s, v in g.items()},
                "grad_ratio": {"R/T": float(g["R"].norm() / (g["T"].norm() + 1e-12)), "R/O": float(g["R"].norm() / (g["O"].norm() + 1e-12))},
                "cos_pair": {a + b: cos(g[a], g[b]) for a, b in itertools.combinations(STREAMS, 2)},
                "cos_with_mixed": {s: cos(v, gm) for s, v in g.items()}}, g

    def kl_of(self, model, step_vec, f):
        m2 = copy.deepcopy(model); self.apply(m2, step_vec)
        idx = torch.arange(0, len(f["obs"]), 8, device="cuda")
        with torch.no_grad():
            d0 = model._dist(f["obs"][idx], f["env"][idx], f["ids"][idx], f["w"][idx]); d1 = m2._dist(f["obs"][idx], f["env"][idx], f["ids"][idx], f["w"][idx])
            return float(gaussian_kl(d0.loc, d0.scale, d1.loc, d1.scale).mean()), m2

    @staticmethod
    def apply(model, vec):
        o = 0
        with torch.no_grad():
            for p in model.actor_parameters():
                n = p.numel(); p.add_(vec[o:o + n].view_as(p)); o += n

    def step_for_kl(self, model, direction, f, target):
        d = direction / (direction.norm() + 1e-12); lo, hi = 0.0, 1.0
        while self.kl_of(model, hi * d, f)[0] < target and hi < 1e3:
            hi *= 2
        for _ in range(30):
            mid = (lo + hi) / 2
            (lo, hi) = (mid, hi) if self.kl_of(model, mid * d, f)[0] < target else (lo, mid)
        kl, m2 = self.kl_of(model, hi * d, f)
        return {"eps": hi, "kl": kl}, m2

    def replay(self, env, model, wv):
        from isaaclab.utils.math import euler_xyz_from_quat
        u = env.unwrapped; robot = u.scene["robot"]; mgr = u.reward_manager; names = list(mgr.active_terms)
        ids = torch.tensor([1, 2], device="cuda").repeat(N, 1); w = torch.tensor(wv, device="cuda").repeat(N, 1)
        res = {k: [] for k in ("F_rate", "F_osc", "F_O", "tl")}
        for es in RESET_SEEDS:
            obs, _ = env.reset(seed=es); th, om, tl, dn = [], [], [], []
            with torch.no_grad():
                for t in range(128):
                    obs, _, te, tr, _ = env.step(model.act_inference(obs["policy"], self.ne(obs), ids, w))
                    rl, pt, _ = euler_xyz_from_quat(robot.data.root_quat_w)
                    wrap = lambda x: torch.atan2(torch.sin(x), torch.cos(x))  # noqa: E731
                    th.append(torch.stack([wrap(rl), wrap(pt)], -1).cpu().numpy()); om.append(robot.data.root_ang_vel_b[:, :2].cpu().numpy())
                    tl.append(mgr._step_reward[:, names.index("track_lin_vel_xy_exp")].cpu().numpy()); dn.append((te | tr).cpu().numpy())
            th, om, tl = np.asarray(th), np.asarray(om), np.asarray(tl); ok = ~np.asarray(dn).any(0)
            ema = np.zeros_like(th); ema[0] = th[0]
            for t in range(1, 128):
                ema[t] = ema[t - 1] + (DT / TAU) * (th[t] - ema[t - 1])
            s = slice(32, 128)
            res["F_rate"].append((om[s][:, ok] ** 2).sum(-1).mean()); res["F_osc"].append(np.sqrt((np.linalg.norm(th[s] - ema[s], axis=-1)[:, ok] ** 2).mean()))
            res["F_O"].append(np.linalg.norm(th[s][:, ok], axis=-1).mean()); res["tl"].append(tl[s][:, ok].mean())
        return {k: float(np.mean(v)) for k, v in res.items()}

    def rollout(self, env, obs) -> dict:
        from talon_rl.models.authority.teacher_v4 import TeacherV4
        ck = torch.load(self.ck, map_location="cuda", weights_only=False)
        model = TeacherV4(num_objectives=3).cuda(); model.load_state_dict(ck["model"]); model.eval()
        self.norm = RunningNormalizer(12, center=True); self.norm.load_state_dict(ck["extrinsics_normalizer"])
        lam_vec = ck.get("lagrange_lambda") or [0.786] * 5
        out = {"schema": "teacher_v4_fca_credit_audit_v1", "read_only": True, "checkpoint": str(self.ck), "lagrange_lambda": lam_vec, "conditions": {}}
        for c, wv in CONDS.items():
            lam = float(lam_vec[region(wv[0])])
            f = self.batch(env, model, wv, lam)
            allm = torch.ones_like(f["loco"])
            row = {"w": wv, "lambda": lam, "loco_fraction": float(f["loco"].float().mean())}
            row["all"], g = self.layers(model, f, allm)
            row["loco"] = self.layers(model, f, f["loco"])[0] if f["loco"].sum() > 100 else None
            if c in STEP_CONDS:
                base = self.replay(env, model, wv); steps = {"baseline": base}
                for name, vec in (("R+", -g["R"]), ("R-", g["R"]), ("mixed+", -sum(g.values()))):
                    for kl in KLS:
                        info, m2 = self.step_for_kl(model, vec, f, kl)
                        r = self.replay(env, m2, wv)
                        steps[f"{name}@{kl}"] = {**info, **r, "rel": {k: r[k] / max(base[k], 1e-12) - 1 for k in ("F_rate", "F_osc", "F_O", "tl")}}
                row["virtual_steps"] = steps
            out["conditions"][c] = row
            L = row["loco"] or row["all"]
            print("COND", c, "lam", round(lam, 2), "loco", round(row["loco_fraction"], 2), "share", {k: round(v, 3) for k, v in L["contribution_share"].items()},
                  "cos", {k: round(v, 2) for k, v in L["cos_pair"].items()}, flush=True)
        self.write(out)
        return out


if __name__ == "__main__":
    a = FCACreditAudit.parse_args((("--seed",), {"type": int, "required": True}), (("--checkpoint",), {"required": True}))
    FCACreditAudit("G1-1", a.seed, 600, a.out, a.checkpoint).execute()
