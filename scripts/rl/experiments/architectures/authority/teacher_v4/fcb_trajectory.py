#!/usr/bin/env python3
"""FC-B R semantic trajectory audit (docs/contracts/teacher_v4/teacher-v4-fcb-r-trajectory-contract.md).

Offline. Checkpoint replay (f2a_bifurcation --traces, every checkpoint of
FB-2a) gives, per checkpoint u and condition, F_rate / F_osc / F_O
(window medians, fc0_audit.windows) and the replay tl. Historical
metrics.jsonl gives lambda and online tl per region. Per seed: the R
semantic state per viable checkpoint (R+ and C both tl >= 0.40), the
lambda phase of the 50 iterations before it, onset of inversion, and the
pre-declared case.
"""
import json
from pathlib import Path

import numpy as np

from fc0_audit import windows

REPO = Path(__file__).resolve().parents[6]
ROOT = REPO / "runs/teacher_v4_fc-2026-09-29/fcb"
RUNS = REPO / "runs/teacher_v4_fb-2026-09-29/fb2a"
SEEDS = (79101, 79102, 79103)
REL, HIGH_LAM, LOW_LAM = 0.05, 3.0, 0.1
REG = {"C": 2, "R+": 3}  # train_v4c LAGR_REGIONS index


def state(dr):
    return "correct" if dr <= -REL else "inverted" if dr >= REL else "flat"


def seed_audit(s):
    rep = json.load(open(ROOT / "traces" / f"seed{s}" / "f2a_replay.json"))["checkpoints"]
    z = np.load(ROOT / "traces" / f"seed{s}" / "rotation_traces.npz")
    hist = [json.loads(l) for l in open(RUNS / f"seed{s}" / "metrics.jsonl")]
    lam = np.array([h["lagrange_lambda"] for h in hist]); tlo = np.array([h["lagrange_tl_online"] for h in hist], dtype=float)
    rows = []
    for u in sorted(int(k) for k in rep):
        med = {}
        for c in ("C", "A+", "O+"):
            W = windows(z[f"{u}|{c}|x"], z[f"{u}|{c}|contact"], z[f"{u}|{c}|ok"])
            med[c] = {k: float(np.median(W[k])) for k in ("F_rate", "F_osc", "F_O")}
            med[c]["tl"] = rep[str(u)][c]["tl"]
        viable = med["A+"]["tl"] >= 0.40 and med["C"]["tl"] >= 0.40
        dR = med["A+"]["F_rate"] / max(med["C"]["F_rate"], 1e-12) - 1
        dO = med["O+"]["F_O"] / max(med["C"]["F_O"], 1e-12) - 1
        win = slice(max(0, u - 50), u)
        lam_w = float(np.mean(lam[win][:, [REG["R+"], REG["C"]]]))
        phase = "high" if lam_w >= HIGH_LAM else "low" if lam_w <= LOW_LAM else "mid"
        rows.append({"it": u, "viable": viable, "dR": dR, "dR_osc": med["A+"]["F_osc"] / max(med["C"]["F_osc"], 1e-12) - 1,
                     "dtl_R": med["A+"]["tl"] - med["C"]["tl"], "dO": dO, "state": state(dR) if viable else "n/a",
                     "lambda_window_RplusC": lam_w, "lambda_phase": phase,
                     "online_tl_window": {k: float(np.nanmean(tlo[win][:, i])) for k, i in REG.items()}, "levels": med})
    V = [r for r in rows if r["viable"]]
    inv = [r["it"] for r in V if r["state"] == "inverted"]
    first_inv = inv[0] if inv else None
    endpoint = [r["state"] for r in V if r["it"] >= 550]
    high = [r["it"] for r in rows if r["lambda_phase"] == "high"]
    if len(V) < 3:
        case = "unresolved (fewer than 3 viable checkpoints)"
    elif first_inv is None:
        case = "no inversion at viable checkpoints"
    else:
        ph = next(r["lambda_phase"] for r in rows if r["it"] == first_inv)
        before_high = high and first_inv < min(high)
        still_inv = bool(endpoint) and endpoint[-1] == "inverted"
        corr_low_after = any(r["state"] == "correct" and r["lambda_phase"] == "low" and r["it"] < first_inv for r in V) if high else False
        if not high:
            case = "C: inverted with no high-lambda phase at all"
        elif before_high:
            case = "C: inversion precedes task dominance"
        elif ph == "high" and still_inv:
            case = "A: inversion onset in the high-lambda phase and persisting (consistent with early task-dominated shaping)"
        elif corr_low_after:
            case = "D: correct at low lambda, later inverted (consistent with later across-update drift)"
        elif ph != "high":
            case = "B: no inversion while lambda was high; inversion after lambda fell (points to accumulated preference updates)"
        else:
            case = "unresolved (inversion in high phase but not persisting)"
    return {"rows": rows, "viable_checkpoints": [r["it"] for r in V], "first_inverted": first_inv, "high_lambda_checkpoints": high,
            "endpoint_states": endpoint, "case": case}


def main():
    out = {"schema": "teacher_v4_fcb_trajectory_v1", "note": "lambda / online tl are historical; replay rows are checkpoint capability, not the training rollout.",
           "seeds": {str(s): seed_audit(s) for s in SEEDS}}
    json.dump(out, open(ROOT / "fcb_trajectory.json", "w"), indent=1)
    for s, v in out["seeds"].items():
        print(s, v["case"])
        cells = []
        for r in v["rows"]:
            d = "" if not r["viable"] else "({:+.2f})".format(r["dR"])
            cells.append("{}:{}{}/{}{:.1f}".format(r["it"], r["state"][:3], d, r["lambda_phase"][0], r["lambda_window_RplusC"]))
        print("   ", " ".join(cells))


if __name__ == "__main__":
    main()
