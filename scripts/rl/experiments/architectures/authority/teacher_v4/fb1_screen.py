#!/usr/bin/env python3
"""FB-1 fixed-task feasibility screen, aggregate (docs/contracts/teacher_v4/teacher-v4-fb1-fixed-task-contract.md).

Offline on f2a_bifurcation replays of FB checkpoints (conditions C, A+ (= R+),
O+, and the two vertices). Per seed: task viability (persistent translation
at R+, C and O+) and preference authority inside feasible locomotion (R+
lowers |w_xy|, O+ lowers tilt, both still translating), then the
pre-declared reading.
"""
import argparse
import json
from pathlib import Path

from f3_screen import persistent

SEEDS = (77101, 77102, 77103)
VIAB = ("A-vertex", "A+", "C", "O+", "O-vertex")  # FB-0 calibrated the whole segment, vertices included
CHECK = ("550", "600")
REL = 0.10


def seed_eval(ck):
    its = sorted(int(k) for k in ck)
    viab = {c: persistent([ck[str(u)][c]["tl"] >= 0.40 for u in its], its) for c in VIAB}
    viable = all(v is not None for v in viab.values())
    auth = {}
    for u in CHECK:
        C, R, O = ck[u]["C"], ck[u]["A+"], ck[u]["O+"]
        auth[u] = {"R_lowers_w_xy": R["w_xy"] <= (1 - REL) * C["w_xy"], "O_lowers_tilt": O["tilt_deg"] <= (1 - REL) * C["tilt_deg"],
                   "R_and_O_translate": R["tl"] >= 0.40 and O["tl"] >= 0.40,
                   "cross": {"R_effect_on_tilt": R["tilt_deg"] / C["tilt_deg"], "O_effect_on_w_xy": O["w_xy"] / C["w_xy"]},
                   "values": {c: {k: ck[u][c][k] for k in ("tl", "w_xy", "tilt_deg", "vert_rms_vz", "class")} for c in ck[u]}}
    authority = all(a["R_lowers_w_xy"] and a["O_lowers_tilt"] and a["R_and_O_translate"] for a in auth.values())
    return {"viability_t_persistent": viab, "viable": viable, "authority": authority, "checkpoints": auth}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--root", required=True)
    p.add_argument("--seeds", default=",".join(map(str, SEEDS)), help="FB-2 uses 78101,78102,78103")
    a = p.parse_args()
    root = Path(a.root); seeds = tuple(int(x) for x in a.seeds.split(","))
    runs = {s: seed_eval(json.load(open(root / f"seed{s}" / "replay" / "f2a_replay.json"))["checkpoints"]) for s in seeds}
    for s, r in runs.items():  # FB-2 dual diagnostics (absent for FB-1)
        rows = [json.loads(l) for l in open(root / f"seed{s}" / "metrics.jsonl")]
        if "lagrange_lambda" in rows[-1]:
            cfg = json.load(open(root / f"seed{s}" / "summary.json"))["lagrange"]
            r["lagrange_final_lambda"] = dict(zip(("O-vertex", "O+", "C", "R+", "R-vertex"), rows[-1]["lagrange_lambda"]))
            r["lagrange_at_cap"] = {k: v >= cfg["cap"] - 1e-6 for k, v in r["lagrange_final_lambda"].items()}
    for r in runs.values():
        r["joint"] = r["viable"] and r["authority"]  # the same policy must show both
    kv = sum(r["viable"] for r in runs.values()); ka = sum(r["authority"] for r in runs.values()); kj = sum(r["joint"] for r in runs.values())
    lag = all("lagrange_at_cap" in r for r in runs.values())
    capped = lag and sum(any(r["lagrange_at_cap"][k] for k in ("R+", "R-vertex")) and not r["viable"] for r in runs.values()) >= 2
    reading = ("PASS: task feasibility and R/O preference semantics coexist in >= 2/3 policies; FB-2 adds V" if kj >= 2
               else ("FAIL viability with R-side duals at cap: R preference may be incompatible with the task under the current realization" if capped
                     else "FAIL viability, duals not at cap: dual adaptation / target insufficient") if lag and kv < 2
               else "FAIL viability: fixed task pressure insufficient; constrained formulation becomes the candidate" if kv < 2
               else "viability without authority: task pressure dominates the preferences; the fixed scalar still has tension" if ka < 2
               else "no joint support: viability and authority occur in different policies")
    out = {"schema": "teacher_v4_fb1_screen_v2", "viable_seeds": kv, "authority_seeds": ka, "joint_seeds": kj, "reading": reading,
           "runs": {str(s): r for s, r in runs.items()}}
    json.dump(out, open(root / "fb1_screen.json", "w"), indent=1)
    print(json.dumps({"viable": kv, "authority": ka, "joint": kj, "reading": reading}, indent=1))


if __name__ == "__main__":
    main()
