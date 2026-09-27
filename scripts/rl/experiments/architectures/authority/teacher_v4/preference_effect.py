#!/usr/bin/env python3
"""V4-C2 null-calibrated preference-effect test (docs/contracts/teacher_v4/teacher-v4-c2-semantic-relation-contract.md, frozen at c1380f3).

Per checkpoint: 512 snapshots (256 envs x env seeds 0,1) after 100 center
warm-up steps, obs corruption off. Per objective i: a discarded center
burn-in branch, then three branches with the treatment w_i+ at a
per-snapshot balanced position ((env + seed) mod 3), each restored in place
and run 32 steps. Per snapshot and H: D = dS_pref - dS_null, with
dS_pref = S(w_i+) - mean of the two centers and dS_null = later center -
earlier center. Simultaneous one-sided 95% bounds over the six H (paired
bootstrap by snapshot, max-t). Gate delta = 0.002; fidelity invalid if any
|mean dS_null| > delta.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))  # scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from g1_evaluate import G1Evaluate, center_w, heavy_w
from rl.core.normalization.running import RunningNormalizer
from twins import restore, semantic_scores, snapshot

N = 256
W = 100
STEPS = 32
HS = (1, 2, 4, 8, 16, 32)
LABELS = ("T", "A", "O", "S")
ENV_SEEDS = (0, 1)
DELTA = 0.002
BOOT = 2000


def simultaneous_bounds(D, rng):
    """D [n, len(HS)] -> per-H mean, one-sided simultaneous 95% LCB and UCB (max-t over H)."""
    n = D.shape[0]; mean = D.mean(0); se = D.std(0, ddof=1) / np.sqrt(n) + 1e-12
    idx = rng.integers(0, n, (BOOT, n))
    bm = D[idx].mean(1)                      # [BOOT, H]
    t = (bm - mean) / se
    c_lo = np.quantile(t.max(1), 0.95)       # for the lower bound
    c_hi = np.quantile((-t).max(1), 0.95)    # for the upper bound
    return mean, mean - c_lo * se, mean + c_hi * se


def classify(lcb, ucb):
    return ["correct" if l > DELTA else ("wrong" if u < -DELTA else "flat") for l, u in zip(lcb, ucb)]


def ladder_class(c):
    cH = dict(zip(HS, c))
    local = cH[1] == "correct" or cH[2] == "correct" or cH[4] == "correct"
    local_all = all(cH[h] == "correct" for h in (1, 2, 4))
    late_ok = cH[16] == "correct" and cH[32] == "correct"
    late_any = cH[16] == "correct" or cH[32] == "correct"
    if not any(x == "correct" for x in c):
        return "no_detectable_semantic_effect"
    if local_all and late_ok:
        return "local_correct_persistent"
    if local and not late_ok:
        return "local_correct_destroyed_by_closed_loop"
    if not local and late_any:
        return "emerges_from_multistep_dynamics"
    return "mixed"


class PreferenceEffect(G1Evaluate):
    report = "preference_effect.json"

    def build_env(self):
        import gymnasium as gym
        import talon_rl.tasks.locomotion.a1_env  # noqa: F401
        from talon_rl.tasks.locomotion.a1_env.v4c_env_cfg import TalonV4CEnvCfg
        cfg = TalonV4CEnvCfg(); cfg.scene.num_envs = N; cfg.seed = 0
        cfg.observations.policy.enable_corruption = False
        self.cfg = cfg
        env = gym.make(self.task, cfg=cfg)
        obs, _ = env.reset(seed=0)
        return env, obs

    def branch(self, env, snap, ids, w):
        u = env.unwrapped
        obs = restore(env, snap, settle=0)
        prev = snap["action"].clone(); S, D = [], []
        with torch.no_grad():
            for _ in range(STEPS):
                a = self.model.act_inference(obs["policy"], self.e_norm(obs), ids, w)
                obs, _, te, tr, _ = env.step(a)
                S.append(semantic_scores(u, a, prev)); prev = a; D.append(te | tr)
        S = torch.stack(S).cpu().numpy(); D = torch.stack(D).cpu().numpy()
        return S, ~np.cumsum(D, 0).astype(bool)

    def rollout(self, env, obs) -> dict:
        from talon_rl.models.authority.teacher_v4 import TeacherV4
        ck = torch.load(self.ck, map_location="cuda", weights_only=False)
        self.model = TeacherV4().cuda(); self.model.load_state_dict(ck["model"]); self.model.eval()
        self.norm = RunningNormalizer(12, center=True); self.norm.load_state_dict(ck["extrinsics_normalizer"])
        ids = torch.arange(4, device="cuda").repeat(N, 1)
        w0 = torch.tensor(center_w(4), device="cuda").repeat(N, 1)
        env_idx = torch.arange(N, device="cuda")
        rng = np.random.default_rng(0)
        D_all = {i: [] for i in range(4)}; null_all = {i: [] for i in range(4)}; excl = {i: {H: 0 for H in HS} for i in range(4)}
        for es in ENV_SEEDS:
            obs, _ = env.reset(seed=es)
            with torch.no_grad():
                for _ in range(W):
                    obs, *_ = env.step(self.model.act_inference(obs["policy"], self.e_norm(obs), ids, w0))
            snap = snapshot(env, obs)
            order = ((env_idx + es) % 3)  # treatment position 0,1,2 among branches 2-4
            for i in range(4):
                wp = torch.tensor(heavy_w(4, i), device="cuda").repeat(N, 1)
                self.branch(env, snap, ids, w0)  # burn-in, discarded
                br = []
                for pos in range(3):
                    w = torch.where((order == pos).unsqueeze(-1), wp, w0)
                    br.append(self.branch(env, snap, ids, w))
                o = order.cpu().numpy(); rows = np.arange(N)
                Sb = np.stack([b[0][..., i] for b in br])       # [3, STEPS, N]
                Ab = np.stack([b[1] for b in br])               # [3, STEPS, N]
                # positions of treatment and of the early/late centers per snapshot
                cen = np.array([[p for p in range(3) if p != oo] for oo in o])  # [N, 2] sorted -> early, late
                Dh, Nh = [], []
                for H in HS:
                    m = Sb[:, :H].mean(1)                        # [3, N]
                    alive = Ab[:, H - 1].all(0)
                    excl[i][H] += int((~alive).sum())
                    tr_ = m[o, rows]; ce = m[cen[:, 0], rows]; cl = m[cen[:, 1], rows]
                    pref = tr_ - 0.5 * (ce + cl); null = cl - ce
                    Dh.append(np.where(alive, pref - null, np.nan)); Nh.append(np.where(alive, null, np.nan))
                D_all[i].append(np.stack(Dh, 1)); null_all[i].append(np.stack(Nh, 1))
        out = {"schema": "teacher_v4_c2_preference_effect_v1", "contract_commit": "c1380f3", "checkpoint": str(self.ck),
               "fold": self.fold, "seed": self.train_seed, "delta": DELTA, "horizons": list(HS), "objectives": {}}
        invalid = False
        for i, lab in enumerate(LABELS):
            D = np.concatenate(D_all[i]); Nn = np.concatenate(null_all[i])
            keep = ~np.isnan(D).any(1)
            mean, lcb, ucb = simultaneous_bounds(D[keep], rng)
            null_mean = np.nanmean(Nn, 0)
            invalid |= bool((np.abs(null_mean) > DELTA).any())
            c = classify(lcb, ucb)
            out["objectives"][lab] = {"n_snapshots": int(keep.sum()), "excluded_terminated": excl[i],
                                      "D_mean": mean.tolist(), "LCB95_sim": lcb.tolist(), "UCB95_sim": ucb.tolist(),
                                      "null_mean": null_mean.tolist(), "class_per_H": c, "ladder": ladder_class(c)}
            print(lab, [f"{x:+.4f}" for x in mean], c, out["objectives"][lab]["ladder"], "null", [f"{x:+.4f}" for x in null_mean], flush=True)
        out["fidelity_invalid"] = invalid
        self.write(out)
        print("FIDELITY_INVALID" if invalid else "FIDELITY_OK", flush=True)
        return out


if __name__ == "__main__":
    a = PreferenceEffect.parse_args((("--fold",), {"choices": ("G1-2", "G1-3"), "required": True}),
                                    (("--seed",), {"type": int, "required": True}))
    PreferenceEffect(a.fold, a.seed, 300, a.out).execute()
