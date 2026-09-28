#!/usr/bin/env python3
"""V4-C3 semantic gates B1/B2 (m=3 anchor), C (seen non-anchor) and D (held-out) for one checkpoint.

Contract: docs/contracts/teacher_v4/teacher-v4-c3-evaluation-contract.md (frozen b258386).
Per group of conditions, from each of 512 center-warm-up snapshots: a
discarded burn-in, then every condition as a 128-step branch restored in
place, positions balanced by a cyclic shift. Conditions are 3-vectors of
weights over (T, A, O); zero weight = inactive (exact padding). Steady window
33-128; transient 1-32 reported. Every group includes the m=3 center C as the
B1 reference.

    anchor : C, T+, A+, O+, and a repeat of one heavy (fidelity null)
    m1     : C, {T}, {A}, {O}
    m2     : C, and for each pair {i,j}: i-heavy (0.7/0.3) and j-heavy

B1: S_i(cond_i) - S_i(C) > delta (one-sided simultaneous over the group's
tests). B2: for each compared pair of conditions, the difference vector over
(T, A, O) has at least one component beyond +-delta (two-sided simultaneous).
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

from g1_evaluate import G1Evaluate
from rl.core.normalization.running import RunningNormalizer
from twins import restore, semantic_scores, snapshot

N = 256
W = 100
STEPS = 128
ENV_SEEDS = (0, 1)
LAB = ("T", "A", "O")
COLS = [0, 1, 2]  # semantic_scores columns T, A, O
BOOT = 2000
FOLDS = {"G1-1": {"seen_other": "m2", "heldout": "m1"}, "G1-2": {"seen_other": "m1", "heldout": "m2"}}


def e(i): v = np.zeros(3); v[i] = 1.0; return v
def heavy3(i): v = np.full(3, 0.15); v[i] = 0.70; return v
def pair_heavy(i, j): v = np.zeros(3); v[i] = 0.70; v[j] = 0.30; return v


GROUPS = {
    "anchor": {"C": np.full(3, 1 / 3), "T+": heavy3(0), "A+": heavy3(1), "O+": heavy3(2)},
    "m1": {"C": np.full(3, 1 / 3), "{T}": e(0), "{A}": e(1), "{O}": e(2)},
    "m2": {"C": np.full(3, 1 / 3), **{f"{LAB[i]}{LAB[j]}:{LAB[k]}": pair_heavy(k, j if k == i else i)
                                      for i, j in itertools.combinations(range(3), 2) for k in (i, j)}},
}
# B1: which condition is objective i's "raised" condition(s); B2: which condition pairs are compared
B1 = {"anchor": {i: [f"{LAB[i]}+"] for i in range(3)},
      "m1": {i: [f"{{{LAB[i]}}}"] for i in range(3)},
      "m2": {i: [f"{LAB[a]}{LAB[b]}:{LAB[i]}" for a, b in itertools.combinations(range(3), 2) if i in (a, b)] for i in range(3)}}
B2 = {"anchor": [("T+", "A+"), ("T+", "O+"), ("A+", "O+")],
      "m1": [("{T}", "{A}"), ("{T}", "{O}"), ("{A}", "{O}")],
      "m2": [(f"{LAB[a]}{LAB[b]}:{LAB[a]}", f"{LAB[a]}{LAB[b]}:{LAB[b]}") for a, b in itertools.combinations(range(3), 2)]}


def sim_bounds(D, rng, two_sided):
    n = D.shape[0]; mean = D.mean(0); se = D.std(0, ddof=1) / np.sqrt(n) + 1e-12
    t = (D[rng.integers(0, n, (BOOT, n))].mean(1) - mean) / se
    if two_sided:
        c = np.quantile(np.abs(t).max(1), 0.95); return mean, mean - c * se, mean + c * se
    return mean, mean - np.quantile(t.max(1), 0.95) * se, mean + np.quantile((-t).max(1), 0.95) * se


class C3Semantics(G1Evaluate):
    report = "c3_semantics.json"

    def __init__(self, fold, seed, delta, out, checkpoint):
        super().__init__(fold, seed, 300, out, checkpoint)
        self.delta = delta

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
                S.append(semantic_scores(u, a, prev)[:, COLS]); prev = a; D.append(te | tr)
        return torch.stack(S).cpu().numpy(), ~torch.stack(D).cpu().numpy().any(0)

    def run_group(self, env, snaps, ids, name):
        conds = GROUPS[name]; keys = list(conds); n_c = len(keys)
        if name == "anchor":
            keys = keys + ["rep"]  # repeat of one heavy, per env
        Wc = torch.tensor(np.stack([conds[k] for k in conds]), dtype=torch.float32, device="cuda")
        env_idx = torch.arange(N, device="cuda")
        per = {k: {"steady": [], "transient": []} for k in conds}; rep = {"steady": [], "transient": []}; excluded = 0
        for es, snap in zip(ENV_SEEDS, snaps):
            P = len(keys)
            slot = (torch.arange(P, device="cuda").unsqueeze(0) + ((env_idx + es) % P).unsqueeze(1)) % P
            rep_c = 1 + env_idx % 3  # the repeated heavy: T+/A+/O+
            content = torch.where(slot == n_c, rep_c.unsqueeze(1), slot) if name == "anchor" else slot
            self.branch(env, snap, ids, Wc[(env_idx % (n_c - 1)) + 1])  # burn-in, discarded
            br = [self.branch(env, snap, ids, Wc[content[:, p]]) for p in range(P)]
            ok = np.all([b[1] for b in br], 0); excluded += int((~ok).sum())
            sl = slot.cpu().numpy(); rows = np.arange(N)
            for wn, win in (("steady", slice(32, 128)), ("transient", slice(0, 32))):
                M = np.stack([b[0][win].mean(0) for b in br])  # [P, N, 3]
                for ci, k in enumerate(conds):
                    per[k][wn].append(M[np.argmax(sl == ci, 1), rows][ok])
                if name == "anchor":
                    rc = rep_c.cpu().numpy()
                    pos = [np.where((sl[n] == n_c) | (sl[n] == rc[n]))[0] for n in range(N)]
                    rep[wn].append(np.stack([M[p[1], n] - M[p[0], n] for n, p in enumerate(pos)])[ok])
        S = {k: {wn: np.concatenate(v[wn]) for wn in v} for k, v in per.items()}
        rng = np.random.default_rng(0); d = self.delta; out = {"excluded_terminated": excluded}
        # interaction matrix (steady): score x condition, relative to C
        out["response_vs_C"] = {k: (S[k]["steady"] - S["C"]["steady"]).mean(0).tolist() for k in conds if k != "C"}
        # B1
        tests = [(i, c) for i in range(3) for c in B1[name][i]]
        D1 = np.stack([S[c]["steady"][:, i] - S["C"]["steady"][:, i] for i, c in tests], 1)
        m1, l1, u1 = sim_bounds(D1, rng, two_sided=False)
        out["B1"] = {f"{LAB[i]}:{c}": {"mean": float(a), "LCB": float(b), "correct": bool(b > d),
                                       "transient_mean": float((S[c]["transient"][:, i] - S["C"]["transient"][:, i]).mean())}
                     for (i, c), a, b in zip(tests, m1, l1)}
        out["B1_pass"] = all(v["correct"] for v in out["B1"].values())
        # B2
        D2 = np.concatenate([S[a]["steady"] - S[b]["steady"] for a, b in B2[name]], 1)  # [n, pairs*3]
        m2, l2, u2 = sim_bounds(D2, rng, two_sided=True)
        out["B2"] = {}
        for p, (a, b) in enumerate(B2[name]):
            comp = {LAB[k]: {"mean": float(m2[3 * p + k]), "LCB": float(l2[3 * p + k]), "UCB": float(u2[3 * p + k])} for k in range(3)}
            dist = any(v["LCB"] > d or v["UCB"] < -d for v in comp.values())
            out["B2"][f"{a} vs {b}"] = {"components": comp, "distinguishable": bool(dist)}
        out["B2_pass"] = all(v["distinguishable"] for v in out["B2"].values())
        out["pass"] = bool(out["B1_pass"] and out["B2_pass"])
        if name == "anchor":
            rm = np.concatenate(rep["steady"]).mean(0)
            out["repeat_null_mean"] = rm.tolist(); out["fidelity_invalid"] = bool((np.abs(rm) > d).any())
            out["ordered_pair_specificity"] = {f"{LAB[i]}{LAB[i]}-{LAB[i]}{LAB[j]}": float((S[f"{LAB[i]}+"]["steady"][:, i] - S[f"{LAB[j]}+"]["steady"][:, i]).mean())
                                               for i in range(3) for j in range(3) if i != j}
        return out

    def rollout(self, env, obs) -> dict:
        from talon_rl.models.authority.teacher_v4 import TeacherV4
        ck = torch.load(self.ck, map_location="cuda", weights_only=False)
        if ck.get("objectives") != "TAO":
            raise SystemExit("c3_semantics expects a V4-C3 (objectives TAO) checkpoint")
        self.model = TeacherV4(num_objectives=3).cuda(); self.model.load_state_dict(ck["model"]); self.model.eval()
        self.norm = RunningNormalizer(12, center=True); self.norm.load_state_dict(ck["extrinsics_normalizer"])
        ids = torch.arange(3, device="cuda").repeat(N, 1)
        w0 = torch.full((N, 3), 1 / 3, device="cuda")
        snaps = []
        for es in ENV_SEEDS:
            obs, _ = env.reset(seed=es)
            with torch.no_grad():
                for _ in range(W):
                    obs, *_ = env.step(self.model.act_inference(obs["policy"], self.e_norm(obs), ids, w0))
            snaps.append(snapshot(env, obs))
        # snapshots are restored in place, so the env state after the loop does not matter
        res = {g: self.run_group(env, snaps, ids, g) for g in ("anchor", "m1", "m2")}
        f = FOLDS[self.fold]
        out = {"schema": "teacher_v4_c3_semantics_v1", "contract": "b258386", "checkpoint": str(self.ck), "fold": self.fold,
               "seed": self.train_seed, "delta": self.delta, "groups": res,
               "B_semantic_pass": res["anchor"]["pass"], "fidelity_invalid": res["anchor"]["fidelity_invalid"],
               "C_seen_group": f["seen_other"], "C_pass": res[f["seen_other"]]["pass"],
               "D_heldout_group": f["heldout"], "D_pass": res[f["heldout"]]["pass"]}
        self.write(out)
        print("C3", self.fold, self.train_seed, {g: (res[g]["B1_pass"], res[g]["B2_pass"]) for g in res}, "fid_invalid", out["fidelity_invalid"], flush=True)
        return out


if __name__ == "__main__":
    a = C3Semantics.parse_args((("--fold",), {"choices": tuple(FOLDS), "required": True}),
                               (("--seed",), {"type": int, "required": True}),
                               (("--checkpoint",), {"required": True}),
                               (("--delta",), {"type": float, "required": True}))
    C3Semantics(a.fold, a.seed, a.delta, a.out, a.checkpoint).execute()
