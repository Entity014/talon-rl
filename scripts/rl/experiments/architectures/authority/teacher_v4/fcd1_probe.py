#!/usr/bin/env python3
"""FC-D1 task-pressure ablation of the FC-C mixed continuation
(docs/contracts/teacher_v4/teacher-v4-fcd1-lambda-zero-contract.md).

Offline. Arm D1 = FC-C arm A with the task stream's actor weight fixed at 0
in every region (--loss-arm pref, mixed sampler). Same u0, run seed and
replay protocol, so D1 and A differ only in the loss. Per seed: the FC-C
endpoint state of D1 (raw + matched Delta_R over k = 30..50) against A's.
"""
import json
from pathlib import Path

import numpy as np

from fcc_probe import ROOT as FCC, U0, arm_audit, selfcheck

ROOT = Path(__file__).resolve().parents[6] / "runs/teacher_v4_fc-2026-09-30/fcd1"


def reading(a, d1):
    if a != "inverted":
        return f"unresolved (arm A not inverted here: {a})"
    if d1 == "inverted":
        return "task pressure in other regions not necessary (remaining: cross-preference updates / dilution)"
    if d1 == "not inverted":
        return "task-dominated updates in other regions contribute"
    return f"unresolved ({d1})"


def main():
    selfcheck()
    out = {"schema": "teacher_v4_fcd1_probe_v1", "seeds": {}}
    for s in U0:
        A, D1 = arm_audit(s, "A", FCC), arm_audit(s, "D1", ROOT)
        lam = [json.loads(l)["lagrange_lambda"] for l in open(FCC / f"seed{s}" / "A" / "metrics.jsonl")]
        out["seeds"][str(s)] = {"A": A, "D1": D1, "u0_dR_D1_minus_A": D1["rows"]["0"]["dR"] - A["rows"]["0"]["dR"],
                                "A_lambda_mean_over_branch": np.mean(lam, 0).tolist(), "reading": reading(A["state"], D1["state"])}
    json.dump(out, open(ROOT / "fcd1_probe.json", "w"), indent=1)
    for s, v in out["seeds"].items():
        print(s, "->", v["reading"], "| u0 check", round(v["u0_dR_D1_minus_A"], 4), "| A lambda", [round(x, 2) for x in v["A_lambda_mean_over_branch"]])
        for arm in ("A", "D1"):
            r = v[arm]
            print("  ", arm, r["state"], "end raw {:+.3f} matched {:+.3f}".format(r["end_dR"], r["end_dR_matched"] if r["end_dR_matched"] is not None else float("nan")),
                  "| shared {:+.0%} |".format(r["shared_change"]), " ".join("{}:{:+.2f}/{:+.2f}/{:+.2f}".format(k, x["dR"], x["dR_matched"], x["dtl"]) for k, x in r["rows"].items()))


if __name__ == "__main__":
    main()
