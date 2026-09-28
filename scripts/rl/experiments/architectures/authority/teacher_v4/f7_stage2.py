#!/usr/bin/env python3
"""F7 stage 2: vertical-axis realization selection (docs/contracts/teacher_v4/teacher-v4-f7-stage2-vertical-realization-contract.md).

Offline on F7 traces. Target per 32-step window: peak-to-peak of the
linearly detrended world root height. Candidates V1-V3 as costs. Rule:
admissibility (|rho| with R and with O < 0.7 per policy) -> behavior-level
Kendall tau -> policy-blocked median within-behavior Spearman -> pooled
Spearman -> stock V1. A choice made only at the last two steps, or with
tau <= 0, is reported as unresolved.
"""
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[6]
TR = REPO / "runs/teacher_v4_f7-2026-09-28/traces"
OUT = REPO / "runs/teacher_v4_f7-2026-09-28/f7_stage2.json"
BEH = {"s73102 C": ("v4c_g1_2_s73102", "C", "s73102"), "s73102 T+": ("v4c_g1_2_s73102", "Tp", "s73102"),
       "F4 74102 T+": ("f4_s74102", "Tp", "s74102"), "F4 74103 T+": ("f4_s74103", "Tp", "s74103"),
       "M0": ("v4c_g1_2_s73102", "M0", "M0")}
CANDS = {"V1 lin_vel_z_l2": ("lin_vel_z_l2", -1.0), "V2 base_lin_acc_z_l2": ("base_lin_acc_z_l2", 1.0),
         "V3 body_height_osc_l2": ("body_height_osc_l2", 1.0)}  # sign turns each into a non-negative cost
RHO_MAX, RHO_TIE = 0.7, 0.02


def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def kendall(x, y):
    n = len(x); s = 0
    for i in range(n):
        for j in range(i + 1, n):
            s += np.sign(x[i] - x[j]) * np.sign(y[i] - y[j])
    return float(s / (n * (n - 1) / 2))


def excursion(z):  # z [96, envs] -> [3 windows * envs]
    W = z.reshape(3, 32, -1); t = np.arange(32)[None, :, None]
    slope = ((t - t.mean()) * (W - W.mean(1, keepdims=True))).sum(1, keepdims=True) / ((t - t.mean()) ** 2).sum()
    d = W - W.mean(1, keepdims=True) - slope * (t - t.mean())
    return (d.max(1) - d.min(1)).ravel()


def main():
    data = {}
    for b, (d, k, pol) in BEH.items():
        z = np.load(TR / d / "substrate_traces.npz"); c = list(z["cols"]); X = z[k][32:128][:, z["ok"]]
        col = lambda n: X[..., c.index(n)]  # noqa: E731
        data[b] = {"policy": pol, "target_win": excursion(col("root_z")),
                   "R_step": -col("ang_vel_xy_l2"), "O_step": -col("flat_orientation_l2"),
                   **{cn: s * col(t) for cn, (t, s) in CANDS.items()}}
    winmean = lambda x: x.reshape(3, 32, -1).mean(1).ravel()  # noqa: E731
    target_means = [float(np.median(data[b]["target_win"])) for b in BEH]
    res = {}
    for cn in CANDS:
        pols = {}
        for b, v in data.items():
            p = pols.setdefault(v["policy"], {"R": [], "O": []})
            p["R"].append(abs(spearman(v[cn].ravel(), v["R_step"].ravel()))); p["O"].append(abs(spearman(v[cn].ravel(), v["O_step"].ravel())))
        rho_R = {p: max(x["R"]) for p, x in pols.items()}; rho_O = {p: max(x["O"]) for p, x in pols.items()}
        admissible = all(x < RHO_MAX for x in rho_R.values()) and all(x < RHO_MAX for x in rho_O.values())
        tau = kendall([float(np.median(winmean(data[b][cn]))) for b in BEH], target_means)
        within = {b: spearman(winmean(data[b][cn]), data[b]["target_win"]) for b in BEH}
        blocked = {}
        for b, r in within.items():
            blocked.setdefault(data[b]["policy"], []).append(r)
        blocked = {p: float(np.median(v)) for p, v in blocked.items()}
        pooled = spearman(np.concatenate([winmean(data[b][cn]) for b in BEH]), np.concatenate([data[b]["target_win"] for b in BEH]))
        res[cn] = {"admissible": admissible, "abs_rho_R_policy": rho_R, "abs_rho_O_policy": rho_O, "tau_behavior": tau,
                   "within_behavior_rho": within, "policy_blocked_rho": blocked, "policy_blocked_median": float(np.median(list(blocked.values()))),
                   "pooled_rho": pooled, "behavior_medians": {b: float(np.median(winmean(data[b][cn]))) for b in BEH}}
    adm = [c for c in CANDS if res[c]["admissible"]]
    decision = {"admissible": adm}
    if not adm:
        decision.update(choice=None, step=None, reading="no admissible V realization; reconsider case-1 / case-2 fallbacks")
    else:
        top = max(res[c]["tau_behavior"] for c in adm); s1 = [c for c in adm if res[c]["tau_behavior"] == top]
        if len(s1) == 1:
            ch, step = s1[0], "tau"
        else:
            m = max(res[c]["policy_blocked_median"] for c in s1); s2 = [c for c in s1 if m - res[c]["policy_blocked_median"] < RHO_TIE]
            if len(s2) == 1:
                ch, step = s2[0], "policy-blocked within-behavior rho"
            else:
                m = max(res[c]["pooled_rho"] for c in s2); s3 = [c for c in s2 if m - res[c]["pooled_rho"] < RHO_TIE]
                ch, step = (s3[0], "pooled rho") if len(s3) == 1 else (next(c for c in CANDS if c in s3), "stock preference")
        unresolved = step in ("pooled rho", "stock preference") or res[ch]["tau_behavior"] <= 0
        decision.update(choice=ch, step=step,
                        reading=("vertical realization unresolved on current selection bank" + (f" (default {ch})" if ch else "")) if unresolved
                        else f"candidate V realization: {ch} (decided at: {step})")
    out = {"schema": "teacher_v4_f7_stage2_v1", "target_behavior_medians_m": dict(zip(BEH, target_means)), "candidates": res, "decision": decision}
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps({"target": out["target_behavior_medians_m"], "decision": decision,
                      "summary": {c: {k: (round(v[k], 3) if isinstance(v[k], float) else v[k]) for k in ("admissible", "tau_behavior", "policy_blocked_median", "pooled_rho")} for c, v in res.items()}}, indent=1))


if __name__ == "__main__":
    main()
