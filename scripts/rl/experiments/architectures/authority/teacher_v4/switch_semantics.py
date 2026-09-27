#!/usr/bin/env python3
"""V4-C2R switch-controlled semantic test (docs/contracts/teacher_v4/teacher-v4-c2r-switch-controlled-contract.md).

Every branch is restored in place from the same center-preference snapshot
and switches at t=0 to a heavy preference, then runs 128 steps. Windows:
transient = steps 1-32, steady = steps 33-128.

--mode null : burn-in heavy (r+1)%4, then three branches all on heavy r
              (r = env % 4). Reports S_i position differences per window.
--mode test : burn-in heavy (r+1)%4, then five branches ordered by a cyclic
              shift of [T+, A+, O+, S+, r+] by (env + seed) % 5.
              D_i = mean_{j != i} [S_i(i+) - S_i(j+)] in the steady window,
              simultaneous one-sided 95% bounds over the four objectives,
              fidelity from the repeated r+ branch. Needs --delta (frozen).
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
STEPS = 128
TRANS = slice(0, 32)
STEADY = slice(32, 128)
LABELS = ("T", "A", "O", "S")
ENV_SEEDS = (0, 1)
BOOT = 2000


def boot_ci(x, rng):
    idx = rng.integers(0, len(x), (BOOT, len(x)))
    m = x[idx].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def simultaneous_bounds(D, rng):
    n = D.shape[0]; mean = D.mean(0); se = D.std(0, ddof=1) / np.sqrt(n) + 1e-12
    bm = D[rng.integers(0, n, (BOOT, n))].mean(1)
    t = (bm - mean) / se
    return mean, mean - np.quantile(t.max(1), 0.95) * se, mean + np.quantile((-t).max(1), 0.95) * se


class SwitchSemantics(G1Evaluate):
    def __init__(self, fold, seed, mode, delta, out):
        self.report = f"switch_semantics_{mode}.json"
        super().__init__(fold, seed, 300, out)
        self.mode, self.delta = mode, delta

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

    def branch(self, env, snap, ids, k_per_env):
        """k_per_env: [N] long, heavy objective index per env."""
        u = env.unwrapped
        W_heavy = torch.tensor(np.stack([heavy_w(4, k) for k in range(4)]), device="cuda")
        w = W_heavy[k_per_env]
        obs = restore(env, snap, settle=0)
        prev = snap["action"].clone(); S, D = [], []
        with torch.no_grad():
            for _ in range(STEPS):
                a = self.model.act_inference(obs["policy"], self.e_norm(obs), ids, w)
                obs, _, te, tr, _ = env.step(a)
                S.append(semantic_scores(u, a, prev)); prev = a; D.append(te | tr)
        S = torch.stack(S).cpu().numpy(); D = torch.stack(D).cpu().numpy()
        return S, ~D.any(0)  # [STEPS, N, 4], [N] survived all 128 steps

    def rollout(self, env, obs) -> dict:
        from talon_rl.models.authority.teacher_v4 import TeacherV4
        ck = torch.load(self.ck, map_location="cuda", weights_only=False)
        self.model = TeacherV4().cuda(); self.model.load_state_dict(ck["model"]); self.model.eval()
        self.norm = RunningNormalizer(12, center=True); self.norm.load_state_dict(ck["extrinsics_normalizer"])
        ids = torch.arange(4, device="cuda").repeat(N, 1)
        w0 = torch.tensor(center_w(4), device="cuda").repeat(N, 1)
        env_idx = torch.arange(N, device="cuda"); r = env_idx % 4
        rng = np.random.default_rng(0)
        acc = []; excluded = 0
        for es in ENV_SEEDS:
            obs, _ = env.reset(seed=es)
            with torch.no_grad():
                for _ in range(W):
                    obs, *_ = env.step(self.model.act_inference(obs["policy"], self.e_norm(obs), ids, w0))
            snap = snapshot(env, obs)
            self.branch(env, snap, ids, (r + 1) % 4)  # burn-in, discarded
            if self.mode == "null":
                br = [self.branch(env, snap, ids, r) for _ in range(3)]
                ok = br[0][1] & br[1][1] & br[2][1]; excluded += int((~ok).sum())
                win = {nm: [b[0][sl].mean(0)[ok] for b in br] for nm, sl in (("transient", TRANS), ("steady", STEADY))}  # [n,4]
                acc.append(win)
            else:
                base = torch.tensor([0, 1, 2, 3, 4], device="cuda")
                shift = (env_idx + es) % 5
                slot = (base.unsqueeze(0) + shift.unsqueeze(1)) % 5          # [N,5]: content index at position p
                content = torch.where(slot == 4, r.unsqueeze(1), slot)       # heavy objective per position
                br = [self.branch(env, snap, ids, content[:, p]) for p in range(5)]
                ok = np.all([b[1] for b in br], 0); excluded += int((~ok).sum())
                sl_np = slot.cpu().numpy(); rows = np.arange(N)
                per = {}
                for nm, sl in (("transient", TRANS), ("steady", STEADY)):
                    M = np.stack([b[0][sl].mean(0) for b in br])             # [5 pos, N, 4]
                    heavy = np.stack([M[np.argmax(sl_np == k, 1), rows] for k in range(4)])  # [4 heavy k, N, 4 score]: first occurrence
                    rep_pos = [np.where((sl_np[n] == 4) | (sl_np[n] == r.cpu().numpy()[n]))[0] for n in range(N)]
                    rep = np.stack([M[rp[1], n] - M[rp[0], n] for n, rp in enumerate(rep_pos)])  # later - earlier, [N,4]
                    per[nm] = (heavy[:, ok], rep[ok])
                acc.append(per)
        out = {"schema": f"teacher_v4_c2r_switch_{self.mode}_v1", "checkpoint": str(self.ck), "fold": self.fold, "seed": self.train_seed,
               "snapshots": N * len(ENV_SEEDS), "excluded_terminated": excluded, "steps": STEPS}
        if self.mode == "null":
            for nm in ("transient", "steady"):
                b = [np.concatenate([a[nm][p] for a in acc]) for p in range(3)]
                out[nm] = {}
                for i, lab in enumerate(LABELS):
                    d21 = b[1][:, i] - b[0][:, i]; d32 = b[2][:, i] - b[1][:, i]
                    out[nm][lab] = {"pos2-pos1": {"mean": float(d21.mean()), "std": float(d21.std()), "ci95": boot_ci(d21, rng)},
                                    "pos3-pos2": {"mean": float(d32.mean()), "std": float(d32.std()), "ci95": boot_ci(d32, rng)}}
                print(nm, {lab: (round(out[nm][lab]["pos2-pos1"]["mean"], 5), round(out[nm][lab]["pos3-pos2"]["mean"], 5), round(out[nm][lab]["pos2-pos1"]["std"], 5)) for lab in LABELS}, flush=True)
        else:
            for nm in ("transient", "steady"):
                heavy = np.concatenate([a[nm][0] for a in acc], 1)   # [4 k, n, 4 score]
                rep = np.concatenate([a[nm][1] for a in acc])        # [n, 4]
                Di = np.stack([np.mean([heavy[i, :, i] - heavy[j, :, i] for j in range(4) if j != i], 0) for i in range(4)], 1)  # [n,4]
                pair = {f"{LABELS[i]}vs{LABELS[j]}": float((heavy[i, :, i] - heavy[j, :, i]).mean()) for i in range(4) for j in range(4) if i != j}
                res = {"D_mean": Di.mean(0).tolist(), "pairwise_mean": pair, "repeat_null_mean": rep.mean(0).tolist()}
                if nm == "steady":
                    mean, lcb, ucb = simultaneous_bounds(Di, rng)
                    res.update({"LCB95_sim": lcb.tolist(), "UCB95_sim": ucb.tolist(),
                                "class": ["correct" if l > self.delta else ("wrong" if u_ < -self.delta else "flat") for l, u_ in zip(lcb, ucb)]})
                    out["fidelity_invalid"] = bool((np.abs(rep.mean(0)) > self.delta).any())
                out[nm] = res
            out["delta"] = self.delta
            print("steady", dict(zip(LABELS, out["steady"]["class"])), [round(x, 4) for x in out["steady"]["D_mean"]],
                  "null", [round(x, 4) for x in out["steady"]["repeat_null_mean"]], "INVALID" if out["fidelity_invalid"] else "OK", flush=True)
        self.write(out)
        return out


if __name__ == "__main__":
    a = SwitchSemantics.parse_args((("--fold",), {"choices": ("G1-2", "G1-3"), "required": True}),
                                   (("--seed",), {"type": int, "required": True}),
                                   (("--mode",), {"choices": ("null", "test"), "required": True}),
                                   (("--delta",), {"type": float, "default": None}))
    if a.mode == "test" and a.delta is None:
        raise SystemExit("--delta (frozen from the null run) is required for --mode test")
    SwitchSemantics(a.fold, a.seed, a.mode, a.delta, a.out).execute()
