#!/usr/bin/env python3
"""F6 conditional raw-feature screen: gentle (L) vs D-costly (H) locomotion (docs/contracts/teacher_v4/teacher-v4-f6-conditional-feature-contract.md).

Offline. Reads substrate_attribution traces (first 64 envs, 128 steps, per-env
survival mask), keeps the steady window 33-128 of surviving envs, and per raw
reward term reports: variance within locomotion, L-vs-H separation on 32-step
windows, behavior-level ordering, Spearman with A per behavior and pooled,
A-conditioned residual separation, motion-vs-standing separation, and the
pre-declared class (complement to A at most). Never selects a D_v2.
"""
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[6]
ROOT = REPO / "runs/teacher_v4_f6-2026-09-28"
DIV = np.array([1.7194554805755615, 0.15590913593769073, 0.01563369482755661])
TERMS = ("track_lin_vel_xy_exp", "track_ang_vel_z_exp", "lin_vel_z_l2", "ang_vel_xy_l2", "dof_torques_l2", "dof_acc_l2",
         "action_rate_l2", "feet_air_time", "flat_orientation_l2")
A = "ang_vel_xy_l2"
# behavior -> (trace dir, npz key, required phenotype)
BEHAVIORS = {
    "L_C": ("v4c_g1_2_s73102", "C", "L"), "L_T+": ("v4c_g1_2_s73102", "Tp", "L"),
    "H_74102": ("f4_s74102", "Tp", "H_D"), "H_74103": ("f4_s74103", "Tp", "H_D"),
    "ref_M0": ("v4c_g1_2_s73102", "M0", "any"), "ref_standing": ("v4c_g1_2_s73101", "C", "standing"),
    "ref_step_74102": ("f4_s74102", "C", "step"), "ref_step_74103": ("f4_s74103", "C", "step"),
}
KEY2COND = {"C": "C", "Tp": "T+", "M0": "M0"}
STRONG, WEAK, RHO_HIGH, NBINS, MIN_WIN = 0.8, 0.5, 0.7, 4, 10


def phenotype(level, gait):
    tl = level["track_lin_vel_xy_exp"]; td = gait["touchdown_step_fraction"]
    R = np.array([level["track_lin_vel_xy_exp"] + level["track_ang_vel_z_exp"], level["ang_vel_xy_l2"], level["flat_orientation_l2"]]) / DIV
    if tl >= 0.40:
        if R[2] < -0.48:
            return "other-costly", R  # O-costly: outside the D question, excluded from the primary screen
        return ("L" if R[1] >= -0.23 else "H_D"), R
    return ("standing" if td < 0.02 else "step"), R


def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def smd(a, b):
    sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) + 1e-12
    return float((a.mean() - b.mean()) / sd)


def residual(wl, al, wh, ah):
    """A-conditioned L-H differences on the common A support: equal-count bins [left, right), last bin closed."""
    lo, hi = max(al.min(), ah.min()), min(al.max(), ah.max())
    inl, inh = (al >= lo) & (al <= hi), (ah >= lo) & (ah <= hi)
    out = {"A_support": [float(lo), float(hi)], "coverage_L": float(inl.mean()), "coverage_H": float(inh.mean()), "bin_deltas": []}
    if lo >= hi:
        return out
    edges = np.quantile(np.concatenate([al[inl], ah[inh]]), np.linspace(0, 1, NBINS + 1))
    for k in range(NBINS):
        last = k == NBINS - 1
        ml = inl & (al >= edges[k]) & ((al <= edges[k + 1]) if last else (al < edges[k + 1]))
        mh = inh & (ah >= edges[k]) & ((ah <= edges[k + 1]) if last else (ah < edges[k + 1]))
        if ml.sum() >= MIN_WIN and mh.sum() >= MIN_WIN:
            out["bin_deltas"].append(float(wl[ml].mean() - wh[mh].mean()))
    return out


def load():
    data, incl = {}, {}
    for name, (d, key, want) in BEHAVIORS.items():
        z = np.load(ROOT / "traces" / d / "substrate_traces.npz")
        j = json.load(open(ROOT / "traces" / d / "substrate_attribution.json"))
        cond = KEY2COND[key]
        ph, R = phenotype(j["level_steady"][cond], j["gait"][cond])
        ok = want == "any" or ph == want
        incl[name] = {"phenotype": ph, "R": R.tolist(), "tl": j["level_steady"][cond]["track_lin_vel_xy_exp"], "included": bool(ok)}
        if ok:
            cols = list(z["cols"]); X = z[key][32:128][:, z["ok"]]  # [96, envs, F]
            data[name] = {t: X[..., cols.index(t)] for t in TERMS}
    return data, incl


def main():
    data, incl = load()
    Lb = [b for b in ("L_C", "L_T+") if b in data]; Hb = [b for b in ("H_74102", "H_74103") if b in data]
    out = {"schema": "teacher_v4_f6_features_v1", "inclusion": incl, "L_behaviors": Lb, "H_behaviors": Hb}
    if not Lb or not Hb:
        out["reading"] = "unresolved: an L or H behavior failed its phenotype inclusion"
        json.dump(out, open(ROOT / "f6_features.json", "w"), indent=1); print(json.dumps(out, indent=1)); return

    def windows(b, t):  # [3 windows x envs] means of 32-step non-overlapping windows
        x = data[b][t]; return x.reshape(3, 32, -1).mean(1).ravel()

    std_b = [b for b in data if b == "ref_standing"]
    feats = {}
    for t in (A, *[x for x in TERMS if x != A]):  # A first: candidates compare against its separation
        wl = np.concatenate([windows(b, t) for b in Lb]); wh = np.concatenate([windows(b, t) for b in Hb])
        s_lh = smd(wl, wh)
        means = {b: float(data[b][t].mean()) for b in data}
        order = (min(means[b] for b in Lb) > max(means[b] for b in Hb)) or (max(means[b] for b in Lb) < min(means[b] for b in Hb))
        rho = {b: spearman(data[b][t].ravel(), data[b][A].ravel()) if t != A else 1.0 for b in Lb + Hb}
        rho_pooled = spearman(np.concatenate([data[b][t].ravel() for b in Lb + Hb]), np.concatenate([data[b][A].ravel() for b in Lb + Hb])) if t != A else 1.0
        rho_policy = {"s73102": max(abs(rho[b]) for b in Lb), **{b.replace("H_", "s"): abs(rho[b]) for b in Hb}}  # L_C and L_T+ are one policy
        al = np.concatenate([windows(b, A) for b in Lb]); ah = np.concatenate([windows(b, A) for b in Hb])
        overall = np.sign(wl.mean() - wh.mean())
        res = residual(wl, al, wh, ah) if t != A else None
        survives = None
        if res is not None and len(res["bin_deltas"]) >= 2:
            n = len(res["bin_deltas"])
            survives = sum(np.sign(d) == overall for d in res["bin_deltas"]) >= max(2, int(np.ceil(0.75 * n)))
        per_h = {}
        for b in Hb:
            if t == A:
                continue
            r = residual(wl, al, windows(b, t), windows(b, A))
            r["direction_compatible"] = None if len(r["bin_deltas"]) < 2 else bool(np.sign(np.mean(r["bin_deltas"])) == overall)
            per_h[b] = r
        compat = [v["direction_compatible"] for v in per_h.values() if v["direction_compatible"] is not None]
        s_motion = smd(np.concatenate([windows(b, t) for b in Lb + Hb]), windows("ref_standing", t)) if std_b else None
        if t == A:
            cls = "baseline (A)"
        elif abs(s_lh) < WEAK:
            cls = "nonspecific / motion feature" if s_motion is not None and abs(s_motion) >= STRONG else "nonspecific"
        elif abs(s_lh) < STRONG or not order:
            cls = "unresolved (weak or conflicting across behaviors)"
        elif survives is None:
            cls = "unresolved (insufficient common A support)"
        elif not survives:
            cls = "redundant with A"
        elif compat and all(compat):
            cls = "candidate complement to A"
        else:
            cls = "unresolved (policy-dependent residual)"
        feats[t] = {"var_within_locomotion": {b: float(data[b][t].var()) for b in Lb + Hb},
                    "S_LH_window": s_lh, "behavior_means": means, "behavior_ordering_consistent": bool(order),
                    "spearman_with_A": rho, "spearman_with_A_pooled": rho_pooled, "abs_rho_policy_level": rho_policy,
                    "residual_pooled": res, "residual_survives": None if survives is None else bool(survives), "residual_per_H_policy": per_h,
                    "S_motion_vs_standing": s_motion, "tracking_term_not_a_D_candidate": t.startswith("track_"), "class": cls}
    out["features"] = feats
    out["note"] = ("L/H are defined with A (R_D), so F6 asks which terms add structure beyond A; it cannot find an alternative to A. "
                   "Gentle side is one policy (s73102). No D_v2 is selected.")
    json.dump(out, open(ROOT / "f6_features.json", "w"), indent=1)
    print(json.dumps({"inclusion": incl, "classes": {t: (round(v["S_LH_window"], 2), v["behavior_ordering_consistent"], v["residual_survives"], v["class"]) for t, v in feats.items()}}, indent=1))


if __name__ == "__main__":
    main()
