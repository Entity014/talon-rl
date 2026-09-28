#!/usr/bin/env python3
"""F8 V realization + controllability screen, aggregate (docs/contracts/teacher_v4/teacher-v4-f8-vertical-controllability-contract.md).

Offline on f2a_bifurcation --conds f8 replays. Per seed, at checkpoints 550
and 600 (both required): task viability, V controllability, R
controllability and R/V double dissociation, on a condition pair set
(primary: T-anchored T55 / T55R / T55V; secondary: C / A+ / V+). Physical
semantics only; no objective returns are compared across arms.
"""
import argparse
import json
from pathlib import Path

import numpy as np

ARMS = ("V1", "V3")
SEEDS = (76101, 76102, 76103)
CHECK = ("550", "600")
REL = 0.10  # a 10 % relative improvement counts as a change
SETS = {"primary": ("T55", "T55R", "T55V"), "secondary": ("C", "A+", "V+")}


def vert(p):
    return np.array([p["vert_rms_vz"], p["vert_excursion_m"]])


def seed_eval(ck, base, rc, vc):
    rows = {}
    for u in CHECK:
        b, r, v = ck[u][base], ck[u][rc], ck[u][vc]
        viable = b["tl"] >= 0.40
        v_ctrl = bool(np.all(vert(v) <= (1 - REL) * vert(b)) and v["tl"] >= 0.40)
        r_ctrl = bool(r["w_xy"] <= (1 - REL) * b["w_xy"] and r["tl"] >= 0.40)
        dissoc = bool(np.all(vert(v) < vert(r)) and r["w_xy"] < v["w_xy"])
        rows[u] = {"viable": bool(viable), "V_controllable": v_ctrl, "R_controllable": r_ctrl, "double_dissociation": dissoc,
                   "pass": bool(viable and v_ctrl and r_ctrl and dissoc),
                   "V_rel_improvement": float(np.mean(1 - vert(v) / np.maximum(vert(b), 1e-9))),
                   "values": {n: {k: p[k] for k in ("tl", "w_xy", "vert_rms_vz", "vert_excursion_m", "tilt_deg", "class")} for n, p in ((base, b), (rc, r), (vc, v))}}
    return {"checkpoints": rows, "pass": all(x["pass"] for x in rows.values()), "viable": all(x["viable"] for x in rows.values()),
            "V_rel_improvement": float(np.mean([x["V_rel_improvement"] for x in rows.values()]))}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--root", required=True); a = p.parse_args()
    root = Path(a.root); out = {"schema": "teacher_v4_f8_screen_v1", "sets": {}}
    for sname, (base, rc, vc) in SETS.items():
        arms = {}
        for arm in ARMS:
            runs = {s: seed_eval(json.load(open(root / f"{arm}_seed{s}" / "replay" / "f2a_replay.json"))["checkpoints"], base, rc, vc) for s in SEEDS}
            k = sum(r["pass"] for r in runs.values()); kv = sum(r["viable"] for r in runs.values())
            arms[arm] = {"pass_count": k, "viable_count": kv, "arm_pass": k >= 2,
                         "median_V_rel_improvement": float(np.median([r["V_rel_improvement"] for r in runs.values()])),
                         "runs": {str(s): r for s, r in runs.items()}}
        passing = [x for x in ARMS if arms[x]["arm_pass"]]
        if len(passing) == 1:
            reading = f"carry {passing[0]} to F9"
        elif len(passing) == 2:
            reading = "both pass: carry both to F9; F8 cannot resolve the realization"
        else:
            nv = {x: arms[x]["viable_count"] for x in ARMS}
            reading = ("task viability unresolved: base mostly non-locomoting; case 3 not rejected" if all(v < 2 for v in nv.values())
                       else "no evidence of a distinct controllable V under this formulation (case 3 not supported at this stage)")
        out["sets"][sname] = {"base_R_V": [base, rc, vc], "arms": arms, "reading": reading}
    json.dump(out, open(root / "f8_screen.json", "w"), indent=1)
    print(json.dumps({s: {"reading": v["reading"], **{arm: (v["arms"][arm]["pass_count"], v["arms"][arm]["viable_count"]) for arm in ARMS}} for s, v in out["sets"].items()}, indent=1))


if __name__ == "__main__":
    main()
