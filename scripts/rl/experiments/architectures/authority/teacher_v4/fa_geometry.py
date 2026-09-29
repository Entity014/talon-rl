#!/usr/bin/env python3
"""FA formulation audit: break-even w_T, normalization sensitivity, locomotion-conditioned Pareto geometry
(docs/contracts/teacher_v4/teacher-v4-fa-formulation-audit-contract.md). Offline, descriptive."""
import itertools
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[6]
SOURCES = {  # trace dir -> conditions to take (M0 only from the s73102 run)
    REPO / "runs/teacher_v4_f7-2026-09-28/traces/v4c_g1_2_s73102": ("C", "T+", "A+", "O+", "M0"),
    REPO / "runs/teacher_v4_f7-2026-09-28/traces/f4_s74102": ("C", "T+", "A+", "O+"),
    REPO / "runs/teacher_v4_f7-2026-09-28/traces/f4_s74103": ("C", "T+", "A+", "O+"),
    REPO / "runs/teacher_v4_fa-2026-09-29/traces/v4c_g1_2_s73101": ("C", "T+", "A+", "O+"),
    **{REPO / f"runs/teacher_v4_fa-2026-09-29/traces/f8_{a}_s{s}": ("C", "T+", "A+", "O+", "V+") for a in ("V1", "V3") for s in (76101, 76102, 76103)},
}
OUT = REPO / "runs/teacher_v4_fa-2026-09-29/fa_geometry.json"
T3B = {"T": 1.7194554805755615, "R": 0.15590913593769073, "O": 0.01563369482755661, "V1": 0.11904645053168333, "V3": 0.0024434582017791437}
FORMS = {"K3": ("T", "R", "O"), "K4-V1": ("T", "R", "V1", "O"), "K4-V3": ("T", "R", "V3", "O")}


def klass(td, tl):
    if tl >= 1.00: return "established"
    if tl >= 0.40: return "partial"
    return "standing" if td < 0.02 else "step-in-place"


def bank():
    rows = []
    for d, conds in SOURCES.items():
        j = json.load(open(d / "substrate_attribution.json"))
        for c in conds:
            L, g = j["level_steady"][c], j["gait"][c]
            comp = {"T": L["track_lin_vel_xy_exp"] + L["track_ang_vel_z_exp"], "R": L["ang_vel_xy_l2"], "O": L["flat_orientation_l2"],
                    "V1": L["lin_vel_z_l2"], "V3": -L["body_height_osc_l2"]}
            rows.append({"name": f"{d.name}:{c}", "tl": L["track_lin_vel_xy_exp"], "td": g["touchdown_step_fraction"],
                         "class": klass(g["touchdown_step_fraction"], L["track_lin_vel_xy_exp"]), "raw": comp})
    return rows


def scaled(rows, scaling):
    keys = ("T", "R", "O", "V1", "V3")
    if scaling == "T3-B":
        div = T3B
    elif scaling == "stock-relative":
        div = {k: 1.0 for k in keys}
    else:  # bank-range
        div = {k: (max(r["raw"][k] for r in rows) - min(r["raw"][k] for r in rows)) or 1.0 for k in keys}
    return np.array([[r["raw"][k] / div[k] for k in keys] for r in rows]), keys


def main():
    rows = bank(); loco = np.array([r["class"] in ("partial", "established") for r in rows])
    out = {"schema": "teacher_v4_fa_geometry_v1", "bank": [{k: r[k] for k in ("name", "tl", "class")} for r in rows],
           "n_locomoting": int(loco.sum()), "n_nonlocomoting": int((~loco).sum()), "results": {}}
    for scaling in ("T3-B", "stock-relative", "bank-range"):
        X, keys = scaled(rows, scaling)
        for fname, comps in FORMS.items():
            idx = [keys.index(c) for c in comps]; Y = X[:, idx]; K = len(comps)
            wts = np.round(np.arange(0, 1.0001, 0.01), 2)
            W = np.array([[wt] + [(1 - wt) / (K - 1)] * (K - 1) for wt in wts])
            J = W @ Y.T                                       # [grid, behaviors]
            best_non = J[:, ~loco].max(1)
            be = {rows[i]["name"]: next((float(wt) for wt, jj, bn in zip(wts, J[:, i], best_non) if jj > bn), None) for i in np.flatnonzero(loco)}
            grid = np.array([w for w in itertools.product(*[np.arange(0, 1.0001, 0.02)] * (K - 1)) if sum(w) <= 1.0 + 1e-9])
            G = np.column_stack([grid, 1 - grid.sum(1)])     # last component takes the remainder
            JG = G @ Y.T
            share = float((JG[:, loco].max(1) > JG[:, ~loco].max(1)).mean())
            vals = [v for v in be.values() if v is not None]
            out["results"][f"{scaling}|{fname}"] = {"break_even_wT": be, "median_break_even_wT": float(np.median(vals)) if vals else None,
                                                   "never_wins": sum(v is None for v in be.values()), "simplex_share_locomotion_wins": share}
    # locomotion-conditioned Pareto geometry (T3-B units; ordering is scale-free per component)
    X, keys = scaled(rows, "T3-B"); Yl = X[loco]; names = [rows[i]["name"] for i in np.flatnonzero(loco)]
    def nondominated(cols):
        Z = Yl[:, [keys.index(c) for c in cols]]
        return [names[i] for i in range(len(Z)) if not any(np.all(Z[j] >= Z[i]) and np.any(Z[j] > Z[i]) for j in range(len(Z)) if j != i)]
    def rho(a, b):
        x, y = Yl[:, keys.index(a)], Yl[:, keys.index(b)]
        rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y)); return float(np.corrcoef(rx, ry)[0, 1])
    out["pareto_locomoting"] = {"R,V1,O": nondominated(("R", "V1", "O")), "R,V3,O": nondominated(("R", "V3", "O")),
                                "T,R,V1,O": nondominated(("T", "R", "V1", "O")), "T,R,V3,O": nondominated(("T", "R", "V3", "O"))}
    out["spearman_locomoting"] = {f"{a}-{b}": rho(a, b) for a, b in (("R", "V1"), ("R", "V3"), ("R", "O"), ("V1", "O"), ("V3", "O"), ("T", "R"), ("T", "V1"), ("T", "V3"), ("T", "O"))}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("bank", len(rows), "locomoting", int(loco.sum()))
    for k, v in out["results"].items():
        print(k.ljust(26), "median w_T*", v["median_break_even_wT"], "never", v["never_wins"], "simplex share %.2f" % v["simplex_share_locomotion_wins"])
    print("locomoting:", [(n.split("/")[-1], round(rows[i]["tl"], 2)) for n, i in zip(names, np.flatnonzero(loco))])
    print(json.dumps({"pareto": out["pareto_locomoting"], "rho": {k: round(v, 2) for k, v in out["spearman_locomoting"].items()}}, indent=1))


if __name__ == "__main__":
    main()
