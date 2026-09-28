#!/usr/bin/env python3
"""F1 locomotion viability of the frozen T/D/O formulation (docs/contracts/teacher_v4/teacher-v4-f1-locomotion-viability-contract.md).

Offline: classifies every substrate-attribution behavior (run, condition) by touchdown
fraction and linear tracking only, scores it with J = w^T [T, D, O] under the
frozen divisors, and maps which class wins over the simplex and the V4-C3
training preference distributions.
"""
import glob
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))

import numpy as np
import torch

from rl.core.algorithms.objective_set_ppo import sample_objective_sets
from talon_rl.rewards.objectives import NORMALIZATION_DIVISORS

OUT = REPO / "runs/teacher_v4_f1-2026-09-28/f1.json"
SOURCES = {"V4-C3": "runs/teacher_v4_c3_substrate-2026-09-28", "V4-C": "runs/teacher_v4_c_substrate-2026-09-28"}
DIV = np.asarray(NORMALIZATION_DIVISORS[:3], np.float64)
CLASSES = ("standing", "step-in-place", "partial", "established")
LOCO = ("partial", "established")
M = 0.05
MARGINS = (0.0, 0.025, 0.05, 0.10)
NAMED = {"C": [1 / 3] * 3, "T+": [.70, .15, .15], "D+": [.15, .70, .15], "O+": [.15, .15, .70],
         "{T}": [1, 0, 0], "{D}": [0, 1, 0], "{O}": [0, 0, 1],
         "TD:T": [.70, .30, 0], "TD:D": [.30, .70, 0], "TO:T": [.70, 0, .30], "TO:O": [.30, 0, .70],
         "DO:D": [0, .70, .30], "DO:O": [0, .30, .70]}
FULL_GATE = ("C", "T+", "D+", "O+")
T_SUBSET_GATE = ("{T}", "TD:T", "TD:D", "TO:T", "TO:O")  # T-containing m=1/m=2 points: must not lose
DISTS = {"m=1": (1,), "m=2": (2,), "m=3": (3,), "G1-1 mix": (2, 3), "G1-2 mix": (1, 3)}


def klass(td, tl):
    if tl >= 1.00: return "established"
    if tl >= 0.40: return "partial"
    return "standing" if td < 0.02 else "step-in-place"


def bank():
    out = []
    for src, d in SOURCES.items():
        fs = sorted(glob.glob(str(REPO / d / "*" / "substrate_attribution.json")))
        if len(fs) != {"V4-C3": 6, "V4-C": 16}[src]:
            sys.exit(f"{src}: expected run count, found {len(fs)}")
        for f in fs:
            j = json.load(open(f))
            for c, L in j["level_steady"].items():
                td, tl = j["gait"][c]["touchdown_step_fraction"], L["track_lin_vel_xy_exp"]
                R = np.array([L["track_lin_vel_xy_exp"] + L["track_ang_vel_z_exp"], L["ang_vel_xy_l2"], L["flat_orientation_l2"]]) / DIV
                out.append({"source": "M0" if c == "M0" else src, "run": f"{j['fold']} s{j['seed']}", "cond": c,
                            "td": td, "tl": tl, "class": klass(td, tl), "R": R.tolist()})
    return out


def best(Rm, W):  # [n_cand,3], [n_w,3] -> max score per w
    return (W @ Rm.T).max(1) if len(Rm) else np.full(len(W), -np.inf)


def main():
    B = bank()
    R = np.array([b["R"] for b in B]); cls = np.array([b["class"] for b in B])
    Rl, Rn = R[np.isin(cls, LOCO)], R[~np.isin(cls, LOCO)]
    grid = np.array([[i / 100, j / 100, 1 - i / 100 - j / 100] for i in range(101) for j in range(101 - i)])
    g = torch.Generator().manual_seed(0)
    dists = {k: sample_objective_sets(100_000, c, g, num_objectives=3)[0].numpy().astype(np.float64) for k, c in DISTS.items()}
    W_named = np.array(list(NAMED.values()), np.float64)

    def margin(W): return best(Rl, W) - best(Rn, W)

    def frac(d, m): return {"win": float((d > m).mean()), "tie": float((np.abs(d) <= m).mean()), "lose": float((d < -m).mean()), "n": int(len(d))}

    def outcome(x, m): return "win" if x > m else "lose" if x < -m else "tie"

    def class_map(W):
        S = np.stack([best(R[cls == c], W) for c in CLASSES], 1)
        wins = np.array(CLASSES)[S.argmax(1)]
        return {c: float((wins == c).mean()) for c in CLASSES}

    dn = margin(W_named); dg = margin(grid); dd = {k: margin(W) for k, W in dists.items()}
    by_margin = {}
    for m in MARGINS:
        named = {k: outcome(x, m) for k, x in zip(NAMED, dn)}
        dist = {}
        for k, W in dists.items():
            hasT = W[:, 0] > 0
            dist[k] = {"all": frac(dd[k], m), "T-containing": frac(dd[k][hasT], m), "T-free": frac(dd[k][~hasT], m) if (~hasT).any() else None}
        full = all(named[k] == "win" for k in FULL_GATE) and all(named[k] != "lose" for k in T_SUBSET_GATE)
        nots = named["C"] == "lose" and all(dist[k]["all"]["win"] < 0.5 for k in ("G1-1 mix", "G1-2 mix"))
        by_margin[str(m)] = {"named": named, "simplex": frac(dg, m), "distributions": dist,
                             "reading": "full support" if full else "not supported" if nots else "region-limited support"}
    named_detail = {k: {"margin_L_minus_N": float(x), "best_per_class": {c: float(best(R[cls == c], np.array([w]))[0]) for c in CLASSES}}
                    for (k, w), x in zip(NAMED.items(), dn)}
    wc = np.array(NAMED["C"]); il = int(np.argmax(Rl @ wc)); gap = float(best(Rn, wc[None])[0] - Rl[il] @ wc + M)
    need = {"best_locomoting_R": Rl[il].tolist(), "score_gap_at_C": gap,
            "needed_D_plus_O_improvement_at_C": gap / (1 / 3) if gap > 0 else 0.0}
    out = {"schema": "teacher_v4_f1_viability_v2", "primary_margin": M, "n_candidates": len(B),
           "class_counts": {s: {c: int(sum(1 for b in B if b["source"] == s and b["class"] == c)) for c in CLASSES} for s in ("V4-C3", "V4-C", "M0")},
           "primary": by_margin[str(M)], "sensitivity": by_margin, "named_detail": named_detail,
           "class_map": {"simplex": class_map(grid), **{k: class_map(W) for k, W in dists.items()}},
           "locomotion_cost_at_C": need, "candidates": B}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("class_counts", "primary", "class_map", "locomotion_cost_at_C")}, indent=1))
    print("readings by margin:", {m: v["reading"] for m, v in by_margin.items()})


if __name__ == "__main__":
    main()
