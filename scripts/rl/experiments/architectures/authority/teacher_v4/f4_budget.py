#!/usr/bin/env python3
"""F4 budget adequacy audit, B0 300 -> 600 (docs/contracts/teacher_v4/teacher-v4-f4-budget-audit-contract.md).

Offline. Joins each seed's F3 B0 replay (checkpoints 50-300) with its
controlled-restart replay (350-600), then applies the pre-declared reading.
"""
import json
from pathlib import Path

import numpy as np

from f3_screen import LOCO, STEPPING, first, persistent

REPO = Path(__file__).resolve().parents[6]
F3 = REPO / "runs/teacher_v4_f3-2026-09-28"
F4 = REPO / "runs/teacher_v4_f4-2026-09-28"
SEEDS = (74101, 74102, 74103)
LR_FLOOR = 1.1e-5


def main():
    runs = {}
    for s in SEEDS:
        ck = {**json.load(open(F3 / f"none_seed{s}/replay/f2a_replay.json"))["checkpoints"],
              **json.load(open(F4 / f"none_seed{s}/replay/f2a_replay.json"))["checkpoints"]}
        its = sorted(int(k) for k in ck)
        r = {"checkpoints": its}
        for c in ("C", "T+"):
            P = [ck[str(u)][c] for u in its]
            flags = [p["class"] in LOCO for p in P]
            r[c] = {"t_step": first([p["class"] in STEPPING for p in P], its), "t_translate": first(flags, its),
                    "t_translate_persistent": persistent(flags, its),
                    "late_locomotion_unresolved": flags[-1] and persistent(flags, its) is None,
                    "tl_by_checkpoint": {u: round(p["tl"], 3) for u, p in zip(its, P)},
                    "class_by_checkpoint": {u: p["class"] for u, p in zip(its, P)}, "final_R": P[-1]["R"]}
        rows = [json.loads(l) for l in open(F4 / f"none_seed{s}/metrics.jsonl")]
        lr = np.array([x["lr"] for x in rows])
        r["lr_floor_fraction"] = float((lr <= LR_FLOOR).mean())
        r["optimizer_stalled"] = r["lr_floor_fraction"] > 0.5
        r["loco_C"] = r["C"]["t_translate_persistent"] is not None
        runs[s] = r
    valid = [s for s in SEEDS if not runs[s]["optimizer_stalled"]]
    k = sum(runs[s]["loco_C"] for s in valid)
    reading = ("invalid: optimizer stalled in a majority of seeds" if len(valid) < 2
               else "budget-sensitive" if k >= 2 else "rare-late" if k == 1 else "no budget signal")
    out = {"schema": "teacher_v4_f4_budget_v1", "runs": {str(s): v for s, v in runs.items()}, "valid_seeds": valid,
           "loco_C": k, "reading": reading}
    json.dump(out, open(F4 / "f4_budget.json", "w"), indent=1)
    print(json.dumps({"reading": reading, "loco_C": k, "valid": valid,
                      "per_seed": {s: {"C": (v["C"]["t_step"], v["C"]["t_translate"], v["C"]["t_translate_persistent"]),
                                       "T+": (v["T+"]["t_step"], v["T+"]["t_translate"], v["T+"]["t_translate_persistent"]),
                                       "lr_floor": v["lr_floor_fraction"]} for s, v in runs.items()}}, indent=1))


if __name__ == "__main__":
    main()
