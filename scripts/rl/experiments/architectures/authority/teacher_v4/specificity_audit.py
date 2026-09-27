#!/usr/bin/env python3
"""V4-C2F data collection for one checkpoint (docs/contracts/teacher_v4/teacher-v4-c2f-specificity-audit-contract.md).

Same snapshot/switch protocol as V4-C2R: 512 snapshots after 100 center
warm-up steps, a discarded burn-in, then five 128-step branches (center and
heavy T/A/O/S) ordered by a cyclic shift of [C,T+,A+,O+,S+] by (env+seed)%5.
Saves an .npz with, per preference, the steady-window per-snapshot mean
scores, 32-step window means, and every 4th per-step score (steady window),
for snapshots with no episode end in any branch. Aggregation is in
specificity_aggregate.py.
"""
from __future__ import annotations

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
ENV_SEEDS = (0, 1)
PREFS = ("C", "T", "A", "O", "S")


class SpecificityAudit(G1Evaluate):
    report = "specificity_audit.json"

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
        return torch.stack(S).cpu().numpy(), ~torch.stack(D).cpu().numpy().any(0)

    def rollout(self, env, obs) -> dict:
        from talon_rl.models.authority.teacher_v4 import TeacherV4
        ck = torch.load(self.ck, map_location="cuda", weights_only=False)
        self.model = TeacherV4().cuda(); self.model.load_state_dict(ck["model"]); self.model.eval()
        self.norm = RunningNormalizer(12, center=True); self.norm.load_state_dict(ck["extrinsics_normalizer"])
        ids = torch.arange(4, device="cuda").repeat(N, 1)
        Wp = torch.tensor(np.stack([center_w(4)] + [heavy_w(4, k) for k in range(4)]), device="cuda")  # [5,4]
        env_idx = torch.arange(N, device="cuda")
        steady, win32, steps = {p: [] for p in PREFS}, {p: [] for p in PREFS}, {p: [] for p in PREFS}
        excluded = 0
        for es in ENV_SEEDS:
            obs, _ = env.reset(seed=es)
            with torch.no_grad():
                for _ in range(W):
                    obs, *_ = env.step(self.model.act_inference(obs["policy"], self.e_norm(obs), ids, Wp[0].repeat(N, 1)))
            snap = snapshot(env, obs)
            self.branch(env, snap, ids, Wp[(env_idx % 4) + 1])  # burn-in on a heavy preference, discarded
            slot = (torch.arange(5, device="cuda").unsqueeze(0) + ((env_idx + es) % 5).unsqueeze(1)) % 5  # [N,5] pref index per position
            br = [self.branch(env, snap, ids, Wp[slot[:, p]]) for p in range(5)]
            ok = np.all([b[1] for b in br], 0); excluded += int((~ok).sum())
            sl = slot.cpu().numpy(); rows = np.arange(N)
            Sall = np.stack([b[0] for b in br])                               # [5 pos, STEPS, N, 4]
            for pi, p in enumerate(PREFS):
                pos = np.argmax(sl == pi, 1)                                  # position of pref pi per env
                S = Sall[pos, :, rows].transpose(1, 0, 2)[:, ok]              # [STEPS, n_ok, 4]
                st = S[32:]
                steady[p].append(st.mean(0))
                win32[p].append(st.reshape(3, 32, -1, 4).mean(1).transpose(1, 0, 2).reshape(-1, 4))
                steps[p].append(st[::4].reshape(-1, 4))
        out = {k: np.concatenate(v) for d, tag in ((steady, "steady"), (win32, "win32"), (steps, "steps")) for k, v in ((f"{tag}_{p}", d[p]) for p in PREFS)}
        np.savez_compressed(self.out / "specificity_audit.npz", **out)
        summary = {"checkpoint": str(self.ck), "fold": self.fold, "seed": self.train_seed, "excluded_terminated": excluded,
                   "n_snapshots": int(len(out["steady_C"])),
                   "M_steady_mean": {p: out[f"steady_{p}"].mean(0).tolist() for p in PREFS}}
        self.write(summary)
        print("M (rows = pref C,T,A,O,S; cols = score T,A,O,S)", {p: [round(x, 4) for x in summary["M_steady_mean"][p]] for p in PREFS}, "excl", excluded, flush=True)
        return summary


if __name__ == "__main__":
    a = SpecificityAudit.parse_args((("--fold",), {"choices": ("G1-2", "G1-3"), "required": True}),
                                    (("--seed",), {"type": int, "required": True}))
    SpecificityAudit(a.fold, a.seed, 300, a.out).execute()
