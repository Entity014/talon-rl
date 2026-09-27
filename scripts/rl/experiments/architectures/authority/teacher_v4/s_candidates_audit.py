#!/usr/bin/env python3
"""V4-C2S data collection for one checkpoint (docs/contracts/teacher_v4/teacher-v4-c2s-smoothness-reformulation-contract.md).

Same snapshot/switch protocol as specificity_audit.py. Per step records
S_T, S_A, S_O and the smoothness candidates S0 (action rate), S1 (action
second difference), S2 (joint-velocity second difference), S3 (applied-torque
rate), all higher-is-better. The first two steps of each branch use the
snapshot's action/velocity history; the steady window (33-128) is unaffected.
Saves per-preference steady-window per-snapshot means, 32-step window means
and every 4th step.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))  # scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from g1_evaluate import center_w, heavy_w, tilt_deg
from specificity_audit import ENV_SEEDS, N, PREFS, STEPS, W, SpecificityAudit
from twins import restore, snapshot

COLS = ("T", "A", "O", "S0", "S1", "S2", "S3")


class SCandidates(SpecificityAudit):
    report = "s_candidates_audit.json"

    def branch(self, env, snap, ids, w):
        u = env.unwrapped
        d = u.scene["robot"].data
        obs = restore(env, snap, settle=0)
        a1 = snap["action"].clone(); a2 = snap["prev_action"].clone()
        v1 = snap["jvel"].clone(); v2 = snap["jvel"].clone()
        t1 = d.applied_torque.clone()
        S, D = [], []
        with torch.no_grad():
            for _ in range(STEPS):
                a = self.model.act_inference(obs["policy"], self.e_norm(obs), ids, w)
                obs, _, te, tr, _ = env.step(a)
                cmd = u.command_manager.get_command("base_velocity")
                v = d.joint_vel; tq = d.applied_torque
                S.append(torch.stack([
                    -((d.root_lin_vel_b[:, 0] - cmd[:, 0]).abs() + (d.root_ang_vel_b[:, 2] - cmd[:, 2]).abs()),
                    -torch.linalg.vector_norm(d.root_ang_vel_b[:, :2], dim=-1),
                    -tilt_deg(d.root_quat_w),
                    -torch.linalg.vector_norm(a - a1, dim=-1),
                    -torch.linalg.vector_norm(a - 2 * a1 + a2, dim=-1),
                    -torch.linalg.vector_norm(v - 2 * v1 + v2, dim=-1),
                    -torch.linalg.vector_norm(tq - t1, dim=-1)], -1))
                a2, a1 = a1, a.clone(); v2, v1 = v1, v.clone(); t1 = tq.clone()
                D.append(te | tr)
        return torch.stack(S).cpu().numpy(), ~torch.stack(D).cpu().numpy().any(0)

    def rollout(self, env, obs) -> dict:
        from talon_rl.models.authority.teacher_v4 import TeacherV4
        from rl.core.normalization.running import RunningNormalizer
        ck = torch.load(self.ck, map_location="cuda", weights_only=False)
        self.model = TeacherV4().cuda(); self.model.load_state_dict(ck["model"]); self.model.eval()
        self.norm = RunningNormalizer(12, center=True); self.norm.load_state_dict(ck["extrinsics_normalizer"])
        ids = torch.arange(4, device="cuda").repeat(N, 1)
        Wp = torch.tensor(np.stack([center_w(4)] + [heavy_w(4, k) for k in range(4)]), device="cuda")
        env_idx = torch.arange(N, device="cuda")
        steady, win32, steps = {p: [] for p in PREFS}, {p: [] for p in PREFS}, {p: [] for p in PREFS}
        excluded = 0
        for es in ENV_SEEDS:
            obs, _ = env.reset(seed=es)
            with torch.no_grad():
                for _ in range(W):
                    obs, *_ = env.step(self.model.act_inference(obs["policy"], self.e_norm(obs), ids, Wp[0].repeat(N, 1)))
            snap = snapshot(env, obs)
            self.branch(env, snap, ids, Wp[(env_idx % 4) + 1])  # burn-in, discarded
            slot = (torch.arange(5, device="cuda").unsqueeze(0) + ((env_idx + es) % 5).unsqueeze(1)) % 5
            br = [self.branch(env, snap, ids, Wp[slot[:, p]]) for p in range(5)]
            ok = np.all([b[1] for b in br], 0); excluded += int((~ok).sum())
            sl = slot.cpu().numpy(); rows = np.arange(N)
            Sall = np.stack([b[0] for b in br])  # [5, STEPS, N, 7]
            for pi, p in enumerate(PREFS):
                S = Sall[np.argmax(sl == pi, 1), :, rows].transpose(1, 0, 2)[:, ok]
                st = S[32:]
                steady[p].append(st.mean(0))
                win32[p].append(st.reshape(3, 32, -1, len(COLS)).mean(1).transpose(1, 0, 2).reshape(-1, len(COLS)))
                steps[p].append(st[::4].reshape(-1, len(COLS)))
        out = {f"{tag}_{p}": np.concatenate(d[p]) for d, tag in ((steady, "steady"), (win32, "win32"), (steps, "steps")) for p in PREFS}
        np.savez_compressed(self.out / "s_candidates_audit.npz", **out)
        summary = {"checkpoint": str(self.ck), "fold": self.fold, "seed": self.train_seed, "excluded_terminated": excluded,
                   "n_snapshots": int(len(out["steady_C"])), "columns": list(COLS)}
        self.write(summary)
        print("done", summary["n_snapshots"], "excl", excluded, flush=True)
        return summary


if __name__ == "__main__":
    a = SCandidates.parse_args((("--fold",), {"choices": ("G1-2", "G1-3"), "required": True}),
                               (("--seed",), {"type": int, "required": True}))
    SCandidates(a.fold, a.seed, 300, a.out).execute()
