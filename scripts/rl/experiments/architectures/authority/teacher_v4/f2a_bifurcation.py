#!/usr/bin/env python3
"""F2-A matched-seed bifurcation of the rare gentle gait (docs/contracts/teacher_v4/teacher-v4-f2a-bifurcation-contract.md).

--mode replay  : one run, every saved checkpoint, conditions C and T+, from a
                 fixed reset suite (same initial states for every checkpoint
                 and run). Writes p_* capability metrics per checkpoint.
--mode analyze : offline; h_* from metrics.jsonl plus the replay outputs ->
                 divergence onsets, probe timestamps, objective-space
                 trajectories, and the pre-declared characterization.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))  # scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

REPO = Path(__file__).resolve().parents[6]
RESET_SEEDS = (910001, 910002)
N = 256
STEADY = slice(32, 128)
CONDS = ("C", "T+")
DIV = np.array([1.7194554805755615, 0.15590913593769073, 0.01563369482755661])
TARGET = ("G1-2", 73102)
PRIMARY = (("G1-2", 73101), ("G1-2", 73103))
SECONDARY = (("G1-3", 73101), ("G1-3", 73102), ("G1-3", 73103))
H_KEYS = ("reward_T", "reward_A", "reward_O", "explained_variance_T", "explained_variance_A", "explained_variance_O",
          "preference_authority", "plant_authority", "log_std_mean", "entropy", "kl", "clip_frac", "surrogate", "value",
          "termination_fraction")
EXPLORATION = ("log_std_mean", "entropy", "kl", "clip_frac")
CREDIT_H = ("explained_variance_T", "explained_variance_A", "explained_variance_O", "value")
CREDIT_P = ("ev_T", "ev_A", "ev_O", "adv_std_T", "adv_std_A", "adv_std_O", "adv_corr_TA", "adv_corr_TO")
RUNLEN = 10


def run_dir(fold, seed):
    return REPO / f"runs/teacher_v4_c_{fold.lower().replace('-', '_')}_seed{seed}-2026-09-27"


def klass(td, tl):
    if tl >= 1.00: return "established"
    if tl >= 0.40: return "partial"
    return "standing" if td < 0.02 else "step-in-place"


# ---------------------------------------------------------------- replay (Isaac)
def replay_main(a):
    import torch
    from g1_evaluate import G1Evaluate, center_w, ev, heavy_w, tilt_deg
    from g1r_critic_evaluate import ROLL, SCORED, mc_targets
    from rl.core.normalization.running import RunningNormalizer

    class Replay(G1Evaluate):
        num_envs = N
        report = "f2a_replay.json"

        def build_env(self):
            import gymnasium as gym
            import talon_rl.tasks.locomotion.a1_env  # noqa: F401
            from talon_rl.tasks.locomotion.a1_env.v4c_env_cfg import TalonV4CEnvCfg
            cfg = TalonV4CEnvCfg(); cfg.scene.num_envs = N; cfg.seed = 0
            cfg.observations.policy.enable_corruption = False
            env = gym.make(self.task, cfg=cfg)
            obs, _ = env.reset(seed=0)
            return env, obs

        def rollout_one(self, env, ids, w):
            from talon_rl.rewards.objectives import normalized_objective_vector
            u = env.unwrapped; robot = u.scene["robot"]; mgr = u.reward_manager
            sensor = u.scene["contact_forces"]; feet = sensor.find_bodies(".*_foot")[0]
            names = list(mgr.active_terms); cols = ["TAOS".index(x) for x in self.labels]
            F, V, R, D = [], [], [], []
            for es in RESET_SEEDS:
                obs, _ = env.reset(seed=es)
                prev = torch.zeros((N, u.action_manager.total_action_dim), device="cuda")
                f_, v_, r_, d_ = [], [], [], []
                with torch.no_grad():
                    for t in range(ROLL):
                        x, e = obs["policy"], self.e_norm(obs)
                        if t < SCORED:
                            v_.append(self.model.query_values(x, e, ids, w, ids).cpu().numpy())
                        act = self.model.act_inference(x, e, ids, w)
                        obs, _, te, tr, _ = env.step(act)
                        raw = mgr._step_reward.detach().cpu().numpy()
                        r_.append(normalized_objective_vector({n: raw[:, i] for i, n in enumerate(names)}, shape=(N,))[:, cols] * u.step_dt)
                        d_.append((te | tr).cpu().numpy())
                        if t < 128:
                            dd = robot.data
                            f_.append(np.column_stack([raw[:, names.index(k)] for k in ("track_lin_vel_xy_exp", "track_ang_vel_z_exp", "ang_vel_xy_l2", "flat_orientation_l2")]
                                                      + [torch.linalg.vector_norm(dd.root_ang_vel_b[:, :2], dim=-1).cpu().numpy(),
                                                         tilt_deg(dd.root_quat_w).cpu().numpy(),
                                                         torch.linalg.vector_norm(dd.joint_acc, dim=-1).cpu().numpy(),
                                                         torch.linalg.vector_norm(dd.joint_vel, dim=-1).cpu().numpy(),
                                                         torch.linalg.vector_norm(act - prev, dim=-1).cpu().numpy(),
                                                         (sensor.data.current_contact_time[:, feet] > 0).float().cpu().numpy()]))
                            prev = act
                F.append(np.asarray(f_)); V.append(np.asarray(v_)); R.append(np.asarray(r_)); D.append(np.asarray(d_))
            F = np.concatenate(F, 1); V = np.concatenate(V, 1); R = np.concatenate(R, 1); D = np.concatenate(D, 1)
            ok = ~D[:128].any(0)
            X = F[STEADY][:, ok]
            ct = X[..., 10:14] > 0.5; prevc = F[31:127][:, ok][..., 10:14] > 0.5
            td = float((ct & ~prevc).any(-1).mean()); tl = float(X[..., 0].mean())
            obj = np.array([X[..., 0].mean() + X[..., 1].mean(), X[..., 2].mean(), X[..., 3].mean()]) / DIV
            G, _, _ = mc_targets(R, D)
            A = G - V
            k = {lab: i for i, lab in enumerate(self.labels)}
            out = {"excluded_terminated": int((~ok).sum()), "td": td, "tl": tl, "class": klass(td, tl), "R": obj.tolist(),
                   "w_xy": float(X[..., 4].mean()), "tilt_deg": float(X[..., 5].mean()), "qdd_norm": float(X[..., 6].mean()),
                   "qd_norm": float(X[..., 7].mean()), "action_rate": float(X[..., 8].mean())}
            for lab in ("T", "A", "O"):
                out[f"ev_{lab}"] = ev(G[..., k[lab]], V[..., k[lab]]); out[f"adv_std_{lab}"] = float(A[..., k[lab]].std())
            out["adv_corr_TA"] = float(np.corrcoef(A[..., k["T"]].ravel(), A[..., k["A"]].ravel())[0, 1])
            out["adv_corr_TO"] = float(np.corrcoef(A[..., k["T"]].ravel(), A[..., k["O"]].ravel())[0, 1])
            return out

        def rollout(self, env, obs):
            from talon_rl.models.authority.teacher_v4 import TeacherV4
            cks = sorted(self.src.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
            res = {}
            for ck_path in cks:
                it = int(ck_path.stem.split("_")[1])
                if it == 0:
                    continue
                ck = torch.load(ck_path, map_location="cuda", weights_only=False)
                self.labels = tuple(ck.get("objectives", "TAOS")); K = len(self.labels)
                self.model = TeacherV4(num_objectives=K).cuda(); self.model.load_state_dict(ck["model"]); self.model.eval()
                self.norm = RunningNormalizer(12, center=True); self.norm.load_state_dict(ck["extrinsics_normalizer"])
                ids = torch.arange(K, device="cuda").repeat(N, 1)
                ws = {"C": center_w(K), "T+": heavy_w(K, 0)}
                res[it] = {c: self.rollout_one(env, ids, torch.tensor(ws[c], device="cuda").repeat(N, 1)) for c in CONDS}
                print("REPLAY", self.fold, self.train_seed, it, {c: (v["class"], round(v["tl"], 3), [round(x, 3) for x in v["R"]]) for c, v in res[it].items()}, flush=True)
            out = {"schema": "teacher_v4_f2a_replay_v1", "source_type": "checkpoint_replay", "historical_claim_allowed": False,
                   "run_dir": str(self.src), "fold": self.fold, "seed": self.train_seed, "reset_seeds": list(RESET_SEEDS),
                   "checkpoints": {str(k): v for k, v in res.items()}}
            self.write(out)
            return out

    r = Replay(a.fold, a.seed, 300, a.out)
    r.src = Path(a.run_dir) if a.run_dir else run_dir(a.fold, a.seed)
    r.execute()


# ---------------------------------------------------------------- analyze (offline)
def history(fold, seed):
    rows = [json.loads(l) for l in open(run_dir(fold, seed) / "metrics.jsonl")]
    h = {k: [] for k in H_KEYS}
    for r in rows:
        for i, lab in enumerate("TAO"):
            h[f"reward_{lab}"].append(r["reward_per_step"][i]); h[f"explained_variance_{lab}"].append(r["explained_variance"][i])
        for k in ("preference_authority", "plant_authority", "entropy", "kl", "clip_frac", "surrogate", "value", "termination_fraction"):
            h[k].append(r[k])
        h["log_std_mean"].append(r["log_std"]["mean"])
    return {k: np.asarray(v, float) for k, v in h.items()}


def onset(x, lo, hi):
    """First iteration (1-based) from which x stays outside [lo, hi] for RUNLEN consecutive iterations."""
    out = (x < lo) | (x > hi)
    for i in range(len(x) - RUNLEN + 1):
        if out[i:i + RUNLEN].all():
            return i + 1
    return None


def first_persistent(flags, its):
    first = next((u for u, f in zip(its, flags) if f), None)
    pers = next((u for j, u in enumerate(its) if all(flags[j:])), None)
    return first, pers


def analyze_main(a):
    src = Path(a.replay_root)
    runs = {"target": [TARGET], "primary": list(PRIMARY), "secondary": list(SECONDARY)}
    rep = {}
    for tier, lst in runs.items():
        for f, s in lst:
            p = src / f"{f.lower().replace('-', '_')}_seed{s}" / "f2a_replay.json"
            rep[(f, s)] = json.load(open(p))["checkpoints"]
    its = sorted(int(k) for k in rep[TARGET])
    # F2-A1 historical: target outside primary-control range for RUNLEN iterations
    H = {k: history(*k) for k in rep}
    h1 = {}
    for key in H_KEYS:
        t = H[TARGET][key]; c = np.stack([H[k][key] for k in PRIMARY])
        h1[key] = {"h_t0_target": float(t[:10].mean()), "h_t0_controls": [float(H[k][key][:10].mean()) for k in PRIMARY],
                   "onset_vs_primary": onset(t, c.min(0), c.max(0)),
                   "onset_vs_primary_and_secondary": onset(t, np.stack([H[k][key] for k in PRIMARY + SECONDARY]).min(0),
                                                           np.stack([H[k][key] for k in PRIMARY + SECONDARY]).max(0))}
    # F2-A2/A3 replay trajectories
    traj = {f"{f} s{s}": {c: [{"it": u, **{k: rep[(f, s)][str(u)][c][k] for k in ("class", "td", "tl", "R", "qdd_norm", "qd_norm", "w_xy", "tilt_deg", *CREDIT_P)},
                               "dR_from_50": (np.array(rep[(f, s)][str(u)][c]["R"]) - np.array(rep[(f, s)][str(its[0])][c]["R"])).tolist()}
                              for u in its] for c in CONDS} for (f, s) in rep}
    # F2-A4 probe timestamps (target)
    ts = {}
    for c in CONDS:
        P = [rep[TARGET][str(u)][c] for u in its]
        props = {"motion": [p["class"] != "standing" for p in P], "contact": [p["td"] >= 0.02 for p in P],
                 "locomotion": [p["class"] in ("partial", "established") for p in P]}
        cred = []
        for u, p in zip(its, P):
            cs = [rep[k][str(u)][c] for k in PRIMARY]
            cred.append(any(p[m] < min(x[m] for x in cs) or p[m] > max(x[m] for x in cs) for m in CREDIT_P))
        props["credit_divergence"] = cred
        ts[c] = {k: dict(zip(("t_probe_first", "t_probe_persistent"), first_persistent(v, its))) for k, v in props.items()}
    # F2-A5 characterization (primary: C)
    loco = ts["C"]["locomotion"]["t_probe_first"]
    if loco is None:
        char = "unresolved: no locomotion at C in any checkpoint"
    elif loco == its[0]:
        char = "unresolved: locomotion at C already at the first checkpoint"
    else:
        cutoff = loco - 50
        expl = [k for k in EXPLORATION if h1[k]["onset_vs_primary"] is not None and h1[k]["onset_vs_primary"] <= cutoff]
        cred_h = [k for k in CREDIT_H if h1[k]["onset_vs_primary"] is not None and h1[k]["onset_vs_primary"] <= cutoff]
        cred_p = ts["C"]["credit_divergence"]["t_probe_first"] is not None and ts["C"]["credit_divergence"]["t_probe_first"] < loco
        e, cr = bool(expl), bool(cred_h or cred_p)
        persistent = ts["C"]["locomotion"]["t_probe_persistent"] == loco
        char = ("mixed" if e and cr else "exploration precursor" if e else "credit precursor" if cr
                else "basin-entry-like" if persistent else "unresolved: no precursor observed and locomotion not persistent")
        char = {"label": char, "locomotion_first_C": loco, "precursor_cutoff_iteration": cutoff,
                "exploration_onsets": expl, "credit_onsets_h": cred_h, "credit_probe_before_locomotion": cred_p}
    order = {"C_locomotion_first": ts["C"]["locomotion"]["t_probe_first"], "Tplus_locomotion_first": ts["T+"]["locomotion"]["t_probe_first"]}
    out = {"schema": "teacher_v4_f2a_v1", "note": "h_* = online_log (historical); p_* = checkpoint_replay (capability only)",
           "F2-A1_historical": h1, "F2-A2_A3_replay_trajectories": traj, "F2-A4_probe_timestamps": ts,
           "C_vs_Tplus_order": order, "F2-A5_characterization": char}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    json.dump(out, open(Path(a.out) / "f2a.json", "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("F2-A4_probe_timestamps", "C_vs_Tplus_order", "F2-A5_characterization")}, indent=1))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("replay", "analyze"), required=True)
    p.add_argument("--fold"); p.add_argument("--seed", type=int); p.add_argument("--run-dir")
    p.add_argument("--replay-root"); p.add_argument("--out", required=True)
    a, rest = p.parse_known_args()
    if a.mode == "analyze":
        analyze_main(a)
    else:
        sys.argv = [sys.argv[0], "--out", a.out] + rest
        replay_main(a)
