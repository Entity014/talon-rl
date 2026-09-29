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
VIAB = ("A+", "C", "O+")
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
    p = argparse.ArgumentParser(); p.add_argument("--root", required=True); a = p.parse_args()
    root = Path(a.root)
    runs = {s: seed_eval(json.load(open(root / f"seed{s}" / "replay" / "f2a_replay.json"))["checkpoints"]) for s in SEEDS}
    kv = sum(r["viable"] for r in runs.values()); ka = sum(r["authority"] for r in runs.values())
    viab, auth = kv >= 2, ka >= 2
    reading = ("PASS: task feasibility and preference authority; abstraction supported, FB-2 adds V" if viab and auth
               else "FAIL viability: fixed task pressure insufficient; constrained formulation becomes the candidate" if not viab
               else "PASS viability, FAIL authority: task pressure dominates the preferences; the fixed scalar still has tension")
    out = {"schema": "teacher_v4_fb1_screen_v1", "viable_seeds": kv, "authority_seeds": ka, "reading": reading, "runs": {str(s): r for s, r in runs.items()}}
    json.dump(out, open(root / "fb1_screen.json", "w"), indent=1)
    print(json.dumps({"viable": kv, "authority": ka, "reading": reading}, indent=1))


if __name__ == "__main__":
    main()
