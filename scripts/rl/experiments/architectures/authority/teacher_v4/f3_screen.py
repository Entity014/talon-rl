#!/usr/bin/env python3
"""F3 preference-invariant locomotion substrate screen: aggregate (docs/contracts/teacher_v4/teacher-v4-f3-substrate-screen-contract.md).

Offline. Reads the F2-A-protocol checkpoint replays of every F3 run
(f2a_bifurcation.py --mode replay) and the runs' metrics.jsonl, and applies
the pre-declared per-arm reading.
"""
import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[6]
ARMS = ("none", "linz", "torque_acc", "air", "all")
LOCO = ("partial", "established")
STEPPING = ("step-in-place", "partial", "established")
D_MIN, O_MIN = -0.23, -0.48  # half of M0's steady-window cost (R_D -0.46, R_O -0.95)


def first(flags, its):
    return next((u for u, f in zip(its, flags) if f), None)


def persistent(flags, its, min_len=2):
    """First checkpoint from which the property holds through the last one, over at least `min_len` checkpoints.
    A property seen only at the final checkpoint cannot be called persistent."""
    return next((u for j, u in enumerate(its) if all(flags[j:]) and len(its) - j >= min_len), None)


def run_summary(replay, metrics):
    ck = replay["checkpoints"]; its = sorted(int(k) for k in ck)
    out = {}
    for c in ("C", "T+"):
        P = [ck[str(u)][c] for u in its]
        out[c] = {"t_contact": first([p["td"] >= 0.02 for p in P], its),
                  "t_step": first([p["class"] in STEPPING for p in P], its),
                  "t_translate": first([p["class"] in LOCO for p in P], its),
                  "t_translate_persistent": persistent([p["class"] in LOCO for p in P], its),
                  "late_locomotion_unresolved": P[-1]["class"] in LOCO and persistent([p["class"] in LOCO for p in P], its) is None,
                  "final": {k: P[-1][k] for k in ("class", "tl", "td", "R", "ev_T", "ev_A", "ev_O")}}
    rows = [json.loads(l) for l in open(metrics)]  # critic EV here is descriptive: with R_shared each head predicts R_i + R_shared
    out["h_last10"] = {"preference_authority": float(np.mean([r["preference_authority"] for r in rows[-10:]])),
                       "shared_reward_per_step": float(np.mean([r.get("shared_reward_per_step", 0.0) for r in rows[-10:]])),
                       "explained_variance": np.mean([r["explained_variance"] for r in rows[-10:]], 0).tolist()}
    out["loco_C"] = out["C"]["t_translate_persistent"] is not None
    R = out["C"]["final"]["R"]
    out["low_cost_C"] = out["loco_C"] and R[1] >= D_MIN and R[2] >= O_MIN
    return out


def arm_reading(runs, b0):
    k = sum(r["loco_C"] for r in runs.values()); kb = sum(r["loco_C"] for r in b0.values())
    positive = k >= 2 and kb <= 1
    lowcost = sum(r["low_cost_C"] for r in runs.values())
    tplus_stand = sum(r["T+"]["final"]["class"] == "standing" for r in runs.values())
    tplus_stand_b0 = sum(r["T+"]["final"]["class"] == "standing" for r in b0.values())
    guard = None if not positive else ("objective-compatible gait" if lowcost >= 2 else "gait discovery improved, objective-compatible gait not recovered")
    return {"n": len(runs), "loco_C": k, "screen_positive": positive, "low_cost_loco_C": lowcost, "guardrail": guard,
            "Tplus_standing_final": tplus_stand, "collapse": k == 0 and tplus_stand > tplus_stand_b0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("screen", "confirm"), required=True)
    p.add_argument("--root", required=True, help="runs/teacher_v4_f3-2026-09-28")
    a = p.parse_args()
    root = Path(a.root)
    seeds = {"screen": (74101, 74102, 74103), "confirm": (74201, 74202, 74203)}[a.stage]
    res = {}
    for arm in ARMS:
        for s in seeds:
            d = root / f"{arm}_seed{s}"
            if (d / "replay" / "f2a_replay.json").exists():
                res.setdefault(arm, {})[s] = run_summary(json.load(open(d / "replay" / "f2a_replay.json")), d / "metrics.jsonl")
    if "none" not in res:
        raise SystemExit("B0 (none) runs are required")
    out = {"schema": "teacher_v4_f3_screen_v1", "stage": a.stage, "runs": {arm: {str(s): v for s, v in r.items()} for arm, r in res.items()},
           "arms": {arm: arm_reading(r, res["none"]) for arm, r in res.items() if arm != "none"},
           "B0": {"loco_C": sum(r["loco_C"] for r in res["none"].values()), "n": len(res["none"])}}
    json.dump(out, open(root / f"f3_{a.stage}.json", "w"), indent=1)
    print(json.dumps({"B0": out["B0"], "arms": out["arms"]}, indent=1))


if __name__ == "__main__":
    main()
