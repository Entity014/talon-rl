#!/usr/bin/env python3
"""FC-0 rotational-stability realization audit (docs/contracts/teacher_v4/teacher-v4-fc0-rotational-realization-audit-contract.md).

Offline on f2a_bifurcation --traces outputs. Per 32-step window (steps
33-128, surviving traced envs) of every locomoting behavior (replay tl >=
0.40): features F_O (posture), F_rate (current R), F_osc, F_acc, F_p2p, and
covariates (command magnitude, speed, cadence, window tl). Staged
classification on the PRIMARY bank (FB-2a) only; the secondary bank is a
descriptive robustness check that never changes a primary class.
"""
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[6]
ROOT = REPO / "runs/teacher_v4_fc-2026-09-29/fc0"
DT, TAU = 0.02, 0.5
PRIMARY = {f"fb2a_s{s}": 600 for s in (79101, 79102, 79103)}
SECONDARY = {**{f"fb2_s{s}": 600 for s in (78101, 78102, 78103)}, "f4_s74102": 600, "f4_s74103": 600, "v4c_s73102": 300}
FEATURES = ("F_rate", "F_osc", "F_acc")          # classified; F_O and F_p2p are reference / descriptive
ETA_MIN, RHO_TASK, RHO_O, NBINS, MIN_GROUP = 0.10, 0.5, 0.7, 4, 5


def windows(tr, contact, ok):
    """tr [128, E, 9], contact [128, E, 4] -> dict of [3*E_ok] window values."""
    tr, contact = tr[:, ok], contact[:, ok] > 0.5
    th, w = tr[..., 0:2], tr[..., 2:4]
    ema = np.zeros_like(th); ema[0] = th[0]
    for t in range(1, 128):
        ema[t] = ema[t - 1] + (DT / TAU) * (th[t] - ema[t - 1])
    dw = np.diff(w, axis=0, prepend=w[:1]) / DT
    td = np.concatenate([np.zeros_like(contact[:1]), contact[1:] & ~contact[:-1]]).any(-1)
    out = {}
    for name, f in (("F_O", lambda s: np.linalg.norm(th[s], axis=-1).mean(0)),
                    ("F_rate", lambda s: (w[s] ** 2).sum(-1).mean(0)),
                    ("F_osc", lambda s: np.sqrt((np.linalg.norm(th[s] - ema[s], axis=-1) ** 2).mean(0))),
                    ("F_acc", lambda s: (dw[s] ** 2).sum(-1).mean(0)),
                    ("F_p2p", lambda s: p2p(th[s])),
                    ("v_cmd", lambda s: np.linalg.norm(tr[s][..., 6:8], axis=-1).mean(0)),
                    ("speed", lambda s: np.linalg.norm(tr[s][..., 4:6], axis=-1).mean(0)),
                    ("cadence", lambda s: td[s].sum(0) / (32 * DT)),
                    ("tl", lambda s: tr[s][..., 8].mean(0))):
        out[name] = np.concatenate([f(slice(32 + 32 * k, 64 + 32 * k)) for k in range(3)])
    return out


def p2p(th):  # th [32, E, 2]: peak-to-peak of linearly detrended roll plus pitch
    t = np.arange(len(th))[:, None, None]
    slope = ((t - t.mean()) * (th - th.mean(0))).sum(0) / ((t - t.mean()) ** 2).sum()
    d = th - th.mean(0) - slope * (t - t.mean())
    return (d.max(0) - d.min(0)).sum(-1)


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def eta2_binned(values, groups, binvar):
    """Mean over equal-count bins of binvar of eta^2(values ~ groups); bins need >= 2 groups with >= MIN_GROUP windows."""
    edges = np.quantile(binvar, np.linspace(0, 1, NBINS + 1)); res = []
    for k in range(NBINS):
        m = (binvar >= edges[k]) & ((binvar <= edges[k + 1]) if k == NBINS - 1 else (binvar < edges[k + 1]))
        v, g = values[m], groups[m]
        keep = [x for x in np.unique(g) if (g == x).sum() >= MIN_GROUP]
        if len(keep) < 2:
            continue
        sel = np.isin(g, keep); v, g = v[sel], g[sel]
        tot = ((v - v.mean()) ** 2).sum()
        if tot <= 0:
            continue
        res.append(float(sum((g == x).sum() * (v[g == x].mean() - v.mean()) ** 2 for x in keep) / tot))
    return float(np.mean(res)) if res else None


def load(bank):
    rows = []
    for name, it in bank.items():
        d = ROOT / "traces" / name
        rep = json.load(open(d / "f2a_replay.json"))["checkpoints"][str(it)]
        z = np.load(d / "rotation_traces.npz")
        for c, v in rep.items():
            if v["tl"] < 0.40:
                continue
            W = windows(z[f"{it}|{c}|x"], z[f"{it}|{c}|contact"], z[f"{it}|{c}|ok"])
            n = len(W["F_O"])
            rows.append({"policy": name, "cond": c, "W": W, "n": n})
    return rows


def classify(rows):
    cat = lambda k: np.concatenate([r["W"][k] for r in rows])  # noqa: E731
    beh = np.concatenate([[f"{r['policy']}/{r['cond']}"] * r["n"] for r in rows])
    pol = np.concatenate([[r["policy"]] * r["n"] for r in rows])
    tl, vcmd, speed, fo = cat("tl"), cat("v_cmd"), cat("speed"), cat("F_O")
    out = {}
    for f in FEATURES + ("F_O", "F_p2p"):
        x = cat(f)
        per_pol = {}
        for p in np.unique(pol):
            m = pol == p
            per_pol[p] = {k: spearman(x[m], v[m]) for k, v in (("v_cmd", vcmd), ("speed", speed), ("cadence", cat("cadence")), ("F_O", fo))}
        med = {k: float(np.median([abs(d[k]) for d in per_pol.values()])) for k in ("v_cmd", "speed", "cadence", "F_O")}
        e_beh, e_pol = eta2_binned(x, beh, tl), eta2_binned(x, pol, tl)
        # task residual: remove a linear fit on [v_cmd, speed], then matched-task eta^2 again
        A = np.column_stack([np.ones_like(x), vcmd, speed]); resid_task = x - A @ np.linalg.lstsq(A, x, rcond=None)[0]
        e_task_resid = eta2_binned(resid_task, beh, tl)
        e_O_cond = eta2_binned(x, beh, fo)  # O-conditioned: bins of posture instead of tracking
        task_assoc = med["v_cmd"] >= RHO_TASK or med["speed"] >= RHO_TASK
        o_assoc = med["F_O"] >= RHO_O
        if f in FEATURES:
            if e_beh is None or e_beh < ETA_MIN:
                cls = "unresolved (not informative at matched task)"
            elif task_assoc and (e_task_resid is None or e_task_resid < ETA_MIN):
                cls = "reject (task-confounded: residual discrimination vanishes)"
            elif o_assoc and (e_O_cond is None or e_O_cond < ETA_MIN):
                cls = "reject (O-redundant: O-conditioned discrimination vanishes)"
            else:
                cls = "candidate" + (" + task-associated flag" if task_assoc else "") + (" + O-associated flag" if o_assoc else "")
        else:
            cls = "reference"
        out[f] = {"class": cls, "median_abs_rho": med, "per_policy_rho": per_pol, "eta2_behavior_matched_tl": e_beh,
                  "eta2_policy_matched_tl": e_pol, "eta2_behavior_task_residual": e_task_resid, "eta2_behavior_O_conditioned": e_O_cond,
                  "task_associated": task_assoc, "O_associated": o_assoc}
    return out


def posture_bias(rows):
    """FB-2a mechanism diagnostic: relative change A+ (R+) vs C per policy, feature vs F_O."""
    res = {}
    for p in sorted({r["policy"] for r in rows}):
        byc = {r["cond"]: r["W"] for r in rows if r["policy"] == p}
        if "A+" in byc and "C" in byc:
            rel = {f: float(np.median(byc["A+"][f]) / max(np.median(byc["C"][f]), 1e-12) - 1) for f in FEATURES + ("F_O",)}
            rel["flags"] = [f for f in FEATURES if rel[f] < 0 and rel["F_O"] > 0]  # improves while posture bias worsens
            res[p] = rel
    return res


def main():
    prim, sec = load(PRIMARY), load(SECONDARY)
    out = {"schema": "teacher_v4_fc0_audit_v1", "primary_behaviors": [f"{r['policy']}/{r['cond']}" for r in prim],
           "secondary_behaviors": [f"{r['policy']}/{r['cond']}" for r in sec],
           "primary": classify(prim), "posture_bias_primary": posture_bias(prim),
           "secondary_descriptive": classify(sec) if sec else None,
           "note": "Primary (FB-2a) decides; secondary never flips a primary class. FC-0 proposes candidates only; FC-1 validates on fresh seeds."}
    json.dump(out, open(ROOT / "fc0_audit.json", "w"), indent=1)
    print(json.dumps({"primary": {f: v["class"] for f, v in out["primary"].items()},
                      "secondary": None if sec is None else {f: v["class"] for f, v in out["secondary_descriptive"].items()},
                      "posture_bias": out["posture_bias_primary"]}, indent=1))


if __name__ == "__main__":
    main()
