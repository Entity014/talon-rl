#!/usr/bin/env python3
"""V4-C3 gate aggregation (docs/contracts/teacher_v4/teacher-v4-c3-evaluation-contract.md, frozen b258386).

Reads c3_semantics.json and anchor_evaluation.json of the six V4-C3 runs and
applies the frozen hierarchy: B = B1 + B2 (anchor, fidelity valid) + B3 (MC256
critic valid for every objective); C = seen non-anchor group; E = B and C in
>= 2/3 runs in both folds; D = held-out group, separate claim. A
fidelity-invalid run is not interpreted and counts as not passing.
"""
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[6] / "runs"
OUT = ROOT / "teacher_v4_c3-2026-09-28" / "aggregate.json"


def load(name):
    fs = sorted(glob.glob(str(ROOT / "teacher_v4_c3_g1_*-2026-09-28" / "**" / name), recursive=True))
    if len(fs) != 6:
        sys.exit(f"expected 6 {name}, found {len(fs)}")
    return {(d["fold"], d["seed"]): d for d in map(lambda f: json.load(open(f)), fs)}


sem, anc = load("c3_semantics.json"), load("anchor_evaluation.json")
runs = {}
for k, s in sem.items():
    valid = not s["fidelity_invalid"]
    b3 = all(v["valid"] for v in anc[k]["critic_mc256"].values())
    runs[f"{k[0]} s{k[1]}"] = {
        "fidelity_invalid": s["fidelity_invalid"],
        "B_semantic": valid and s["B_semantic_pass"], "B3_critic": b3,
        "critic_valid": {o: v["valid"] for o, v in anc[k]["critic_mc256"].items()},
        "B": valid and s["B_semantic_pass"] and b3,
        "C": valid and s["C_pass"], "D": valid and s["D_pass"],
        "B2_all_groups": all(g["B2_pass"] for g in s["groups"].values()),
    }
fold = {}
for f in ("G1-1", "G1-2"):
    rs = [v for k, v in runs.items() if k.startswith(f)]
    fold[f] = {g: sum(r[g] for r in rs) for g in ("B_semantic", "B3_critic", "B", "C", "D")}
    fold[f]["B_and_C"] = sum(r["B"] and r["C"] for r in rs)
out = {"runs": runs, "folds": fold,
       "objective_layer_valid": all(v["B_and_C"] >= 2 for v in fold.values()),
       "heldout_generalization": all(v["D"] >= 2 for v in fold.values())}
# post-hoc sensitivity (contract): G1-1 s73102 excluded, 2 runs left in G1-1
out["sensitivity_without_G1-1_s73102_B_and_C"] = sum(v["B"] and v["C"] for k, v in runs.items() if k.startswith("G1-1") and k != "G1-1 s73102")
OUT.parent.mkdir(parents=True, exist_ok=True)
json.dump(out, open(OUT, "w"), indent=2)
print(json.dumps(out["folds"]), "valid", out["objective_layer_valid"], "heldout", out["heldout_generalization"])
