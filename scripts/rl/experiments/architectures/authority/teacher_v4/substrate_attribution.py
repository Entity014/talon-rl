#!/usr/bin/env python3
"""V4-C3 substrate attribution, H2 read-only (docs/contracts/teacher_v4/teacher-v4-c3-substrate-attribution-contract.md).

Anchor-group switch-controlled branches C, T+, A+, O+ and M0 from the same
512 snapshots, positions balanced by a cyclic shift. Logs every reward term
and gait/physics channels per step; reports substrate level (C - M0) and
response (j+ - C) with simultaneous bounds, gait structure, and
phase-conditioned |w_xy|^2.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))  # scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from c3_semantics import ENV_SEEDS, N, STEPS, W, C3Semantics, heavy3
from g1_evaluate import tilt_deg
from rl.core.normalization.running import RunningNormalizer
from twins import restore, semantic_scores, snapshot

M0_CKPT = Path(__file__).resolve().parents[6] / "runs/m0_1_seed0_2026-09-22/model_299.pt"
CONDS = ("C", "T+", "A+", "O+", "M0")
SUBSTRATE = ("lin_vel_z_l2", "dof_torques_l2", "dof_acc_l2", "feet_air_time")
STEADY = slice(32, 128)
MATERIAL = 0.10
TRACE_ENVS = 64
BOOT = 2000


def sim_bounds_2s(D, rng):
    n = D.shape[0]; mean = D.mean(0); se = D.std(0, ddof=1) / np.sqrt(n) + 1e-12
    t = np.abs((D[rng.integers(0, n, (BOOT, n))].mean(1) - mean) / se).max(1)
    c = np.quantile(t, 0.95); return mean, mean - c * se, mean + c * se


class SubstrateAttribution(C3Semantics):
    report = "substrate_attribution.json"

    def __init__(self, fold, seed, out, checkpoint):
        super().__init__(fold, seed, 0.0, out, checkpoint)

    def branch(self, env, snap, ids, w, is_m0):
        u = env.unwrapped; robot = u.scene["robot"]; mgr = u.reward_manager
        sensor = u.scene["contact_forces"]; feet = sensor.find_bodies(".*_foot")[0]
        obs = restore(env, snap, settle=0)
        prev = snap["action"].clone(); F, D = [], []
        with torch.no_grad():
            for _ in range(STEPS):
                a = self.model.act_inference(obs["policy"], self.e_norm(obs), ids, w)
                a0 = self.m0.act_inference_with_preference(obs["policy"], self.w_m0).clamp(-1, 1)
                a = torch.where(is_m0.unsqueeze(1), a0, a)
                obs, _, te, tr, _ = env.step(a)
                d = robot.data
                contact = (sensor.data.current_contact_time[:, feet] > 0).float()
                F.append(torch.cat([mgr._step_reward, semantic_scores(u, a, prev)[:, :3],
                                    torch.stack([torch.linalg.vector_norm(d.root_ang_vel_b[:, :2], dim=-1), tilt_deg(d.root_quat_w),
                                                 d.root_lin_vel_b[:, 2], torch.linalg.vector_norm(d.joint_acc, dim=-1),
                                                 torch.linalg.vector_norm(d.applied_torque, dim=-1)], -1), contact], -1))
                prev = a; D.append(te | tr)
        return torch.stack(F).cpu().numpy(), ~torch.stack(D).cpu().numpy().any(0)

    def rollout(self, env, obs) -> dict:
        from talon_rl.models.authority.teacher_v4 import TeacherV4
        from talon_rl.models.foundations.three_objective import V1CSharedActorCritic, initialize_from_rsl_m01
        ck = torch.load(self.ck, map_location="cuda", weights_only=False)
        if ck.get("objectives") != "TAO":
            raise SystemExit("substrate_attribution expects a V4-C3 (objectives TAO) checkpoint")
        self.model = TeacherV4(num_objectives=3).cuda(); self.model.load_state_dict(ck["model"]); self.model.eval()
        self.norm = RunningNormalizer(12, center=True); self.norm.load_state_dict(ck["extrinsics_normalizer"])
        u = env.unwrapped
        self.m0 = V1CSharedActorCritic(obs["policy"].shape[-1], u.action_manager.total_action_dim).cuda()
        initialize_from_rsl_m01(self.m0, M0_CKPT, device="cpu"); self.m0.cuda().eval()
        self.w_m0 = torch.tensor([1., 0., 0.], device="cuda").repeat(N, 1)
        names = list(u.reward_manager.active_terms)
        cols = names + ["S_T", "S_A", "S_O", "w_xy", "tilt_deg", "v_z", "qdd_norm", "tau_norm", "c_FL", "c_FR", "c_RL", "c_RR"]
        ids = torch.arange(3, device="cuda").repeat(N, 1)
        w0 = torch.full((N, 3), 1 / 3, device="cuda")
        Wc = torch.tensor(np.stack([np.full(3, 1 / 3), heavy3(0), heavy3(1), heavy3(2), np.full(3, 1 / 3)]), dtype=torch.float32, device="cuda")
        env_idx = torch.arange(N, device="cuda"); P = len(CONDS)
        per = {c: [] for c in CONDS}; traces = {c: [] for c in CONDS}; excluded = 0
        for es in ENV_SEEDS:
            obs, _ = env.reset(seed=es)
            with torch.no_grad():
                for _ in range(W):
                    obs, *_ = env.step(self.model.act_inference(obs["policy"], self.e_norm(obs), ids, w0))
            snap = snapshot(env, obs)
            no_m0 = torch.zeros(N, dtype=torch.bool, device="cuda")
            self.branch(env, snap, ids, Wc[1 + env_idx % 3], no_m0)  # burn-in, discarded
            slot = (torch.arange(P, device="cuda").unsqueeze(0) + ((env_idx + es) % P).unsqueeze(1)) % P
            br = [self.branch(env, snap, ids, Wc[slot[:, p]], slot[:, p] == 4) for p in range(P)]
            ok = np.all([b[1] for b in br], 0); excluded += int((~ok).sum())
            sl = slot.cpu().numpy(); rows = np.arange(N)
            X = np.stack([b[0] for b in br])  # [P, STEPS, N, F]
            for ci, c in enumerate(CONDS):
                pos = np.argmax(sl == ci, 1)
                Xc = X[pos, :, rows].transpose(1, 0, 2)  # [STEPS, N, F]
                per[c].append(Xc[:, ok]); traces[c].append(Xc[:, :TRACE_ENVS])
        S = {c: np.concatenate(v, 1) for c, v in per.items()}  # [STEPS, n, F]
        M = {c: S[c][STEADY].mean(0) for c in CONDS}           # [n, F]
        rng = np.random.default_rng(0); si = [names.index(k) for k in SUBSTRATE]
        # gated contrasts: level C - M0 and responses j+ - C, over the substrate terms
        contr = {"C-M0": ("C", "M0"), "T+-C": ("T+", "C"), "A+-C": ("A+", "C"), "O+-C": ("O+", "C")}
        Dm = np.concatenate([M[a][:, si] - M[b][:, si] for a, b in contr.values()], 1)
        mean, lo, hi = sim_bounds_2s(Dm, rng)
        m0_level = np.abs(M["M0"][:, si].mean(0))
        gated = {}
        for ci, cname in enumerate(contr):
            gated[cname] = {}
            for k, term in enumerate(SUBSTRATE):
                j = ci * len(SUBSTRATE) + k
                mat = bool((lo[j] > 0 or hi[j] < 0) and abs(mean[j]) >= MATERIAL * m0_level[k])
                gated[cname][term] = {"mean": float(mean[j]), "lo": float(lo[j]), "hi": float(hi[j]),
                                      "material": mat, "material_negative": bool(mat and mean[j] < 0)}
        s1 = sum(v["material_negative"] for v in gated["C-M0"].values()) >= 2
        s2 = any(v["material_negative"] for v in gated["A+-C"].values())
        # descriptive
        level = {c: dict(zip(cols, M[c].mean(0).tolist())) for c in CONDS}
        delta = {c: dict(zip(cols, (M[c] - M["C"]).mean(0).tolist())) for c in CONDS if c != "C"}
        gait = {}
        wi = cols.index("w_xy"); ci0 = cols.index("c_FL")
        for c in CONDS:
            X = S[c][STEADY]; ct = X[..., ci0:ci0 + 4] > 0.5  # FL FR RL RR
            n_contact = ct.sum(-1)
            diag = (ct[..., 0] & ct[..., 3] & ~ct[..., 1] & ~ct[..., 2]) | (ct[..., 1] & ct[..., 2] & ~ct[..., 0] & ~ct[..., 3])
            prevc = S[c][31:127, :, ci0:ci0 + 4] > 0.5
            td = (ct & ~prevc).any(-1)
            w2 = X[..., wi] ** 2
            gait[c] = {"contact_count_hist": [float((n_contact == k).mean()) for k in range(5)],
                       "diagonal_pair_fraction": float(diag.mean()), "touchdown_step_fraction": float(td.mean()),
                       "w_xy2_touchdown": float(w2[td].mean()) if td.any() else None,
                       "w_xy2_other": float(w2[~td].mean()) if (~td).any() else None}
        out = {"schema": "teacher_v4_c3_substrate_attribution_v1", "checkpoint": str(self.ck), "fold": self.fold,
               "seed": self.train_seed, "m0_checkpoint": str(M0_CKPT), "excluded_terminated": excluded,
               "n_snapshots_used": int(M["C"].shape[0]), "gated": gated, "S1_off_manifold": bool(s1),
               "S2_D_degrades_substrate": bool(s2), "level_steady": level, "delta_vs_C_steady": delta, "gait": gait}
        np.savez_compressed(self.out / "substrate_traces.npz", cols=np.asarray(cols),
                            **{c.replace("+", "p"): np.concatenate(v, 1).astype(np.float32) for c, v in traces.items()})
        self.write(out)
        print("SUB", self.fold, self.train_seed, "S1", s1, "S2", s2,
              {c: {t: round(v["mean"], 4) for t, v in g.items()} for c, g in gated.items()}, flush=True)
        return out


if __name__ == "__main__":
    a = SubstrateAttribution.parse_args((("--fold",), {"choices": ("G1-1", "G1-2"), "required": True}),
                                        (("--seed",), {"type": int, "required": True}),
                                        (("--checkpoint",), {"required": True}))
    SubstrateAttribution(a.fold, a.seed, a.out, a.checkpoint).execute()
