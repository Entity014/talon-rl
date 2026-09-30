#!/usr/bin/env python3
"""FC-D2 R-vertex sufficiency test (docs/contracts/teacher_v4/teacher-v4-fcd2-rvertex-sufficiency-contract.md).

Offline. D2 = FC-C arm A with only the R-vertex lambda in the actor loss
(--lambda-regions R-vertex); D2p (79101 only, secondary) = only the R+
lambda. Same u0, run seed and replay protocol as A. Primary: 79101, the
only seed where A carried task pressure outside the R vertex. In 79102 and
79103, A's lambda outside the R vertex was ~0, so D2 there is a near-repeat
of A (consistency check).
"""
import json
from pathlib import Path

import numpy as np

from fcc_probe import ROOT as FCC, U0, arm_audit, selfcheck

ROOT = Path(__file__).resolve().parents[6] / "runs/teacher_v4_fc-2026-09-30/fcd2"
PRIMARY = 79101


def lam_mean(d):
    return np.mean([json.loads(l)["lagrange_lambda"] for l in open(d / "metrics.jsonl")], 0).tolist()


def main():
    selfcheck()
    out = {"schema": "teacher_v4_fcd2_probe_v1", "seeds": {}}
    for s in U0:
        A, D2 = arm_audit(s, "A", FCC), arm_audit(s, "D2", ROOT)
        row = {"A": A, "D2": D2, "u0_dR_D2_minus_A": D2["rows"]["0"]["dR"] - A["rows"]["0"]["dR"],
               "lambda_mean": {"A": lam_mean(FCC / f"seed{s}" / "A"), "D2": lam_mean(ROOT / f"seed{s}" / "D2")}}
        if s == PRIMARY:
            row["D2p"] = arm_audit(s, "D2p", ROOT); row["lambda_mean"]["D2p"] = lam_mean(ROOT / f"seed{s}" / "D2p")
        out["seeds"][str(s)] = row
    p, reps = out["seeds"][str(PRIMARY)], [out["seeds"][str(s)] for s in U0 if s != PRIMARY]
    if p["A"]["state"] != "inverted":
        v = "unresolved (primary A not inverted)"
    elif p["D2"]["state"] == "inverted":
        v = "R-vertex task pressure sufficient (within mixed training)" + ("" if all(r["D2"]["state"] == "inverted" for r in reps)
                                                                          else "; near-repeat seeds not all inverted (branch-sensitive)")
    elif p["D2"]["state"] == "not inverted":
        v = "R-vertex task pressure not sufficient alone in 79101 (needs pressure in other regions or A's dual interaction)"
    else:
        v = f"unresolved ({p['D2']['state']})"
    out["verdict"] = v
    json.dump(out, open(ROOT / "fcd2_probe.json", "w"), indent=1)
    print("VERDICT", v)
    for s, row in out["seeds"].items():
        print(s, "u0 check", round(row["u0_dR_D2_minus_A"], 4), "| lambda", {k: [round(x, 2) for x in l] for k, l in row["lambda_mean"].items()})
        for arm in ("A", "D2", "D2p"):
            if arm in row:
                r = row[arm]
                print("  ", arm, r["state"], "end raw {:+.3f} matched {:+.3f}".format(r["end_dR"], r["end_dR_matched"] if r["end_dR_matched"] is not None else float("nan")),
                      "|", " ".join("{}:{:+.2f}/{:+.2f}/{:+.2f}".format(k, x["dR"], x["dR_matched"], x["dtl"]) for k, x in r["rows"].items()))


if __name__ == "__main__":
    main()
