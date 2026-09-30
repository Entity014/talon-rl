#!/usr/bin/env python3
"""FC-G preregistered paired endpoint screen. Run after trace replay."""
import argparse
import json
from pathlib import Path

import numpy as np

from fcc_probe import END, arm_audit, selfcheck

ROOT = Path(__file__).resolve().parents[6] / "runs/teacher_v4_fc-2026-10-01/fcg"
SEEDS = (79101, 79103)
ARMS = ("SPLIT", "SPLIT-NORM-MATCHED")
REPEATS = {1: (11, 12, 13), 2: (11, 12, 13, 14, 15)}


def classify(pairs, stage):
    effects = [p["matched_pp"] for p in pairs if p["valid"]]
    med = float(np.median(effects)) if effects else None
    need = 3 if stage == 1 else 4
    if len(effects) >= need and sum(x > 0 for x in effects) >= need and med > 10:
        return "replicated attenuation", med
    if len(effects) >= need and sum(x < 0 for x in effects) >= need and med < -10:
        return "replicated worsening", med
    if len(effects) >= need and sum(abs(x) < 10 for x in effects) >= need and abs(med) < 5:
        return "within practical band", med
    return ("open" if stage == 1 else "not resolved by registered budget"), med


def pair(seed, arm, repeat):
    base = arm_audit(seed, f"GLOBAL_r{repeat}", ROOT)
    candidate = arm_audit(seed, f"{arm}_r{repeat}", ROOT)
    valid = all(v["end_dR_matched"] is not None and not v["state"].startswith(("collapse", "unresolved"))
                for v in (base, candidate))
    return {"r": repeat, "valid": valid, "states": [base["state"], candidate["state"]],
            "endpoint_gates": [endpoint_gates(v) for v in (base, candidate)],
            "matched_pp": 100 * (base["end_dR_matched"] - candidate["end_dR_matched"]) if valid else None,
            "raw_pp": 100 * (base["end_dR"] - candidate["end_dR"]) if valid else None}


def endpoint_gates(audit):
    viable = [audit["rows"][str(k)] for k in END if audit["rows"][str(k)]["viable"]]
    return {"viable": len(viable) >= 2,
            "r_restored": len(viable) >= 2 and audit["end_dR"] <= -0.05 and
                          audit["end_dR_matched"] is not None and audit["end_dR_matched"] <= -0.05,
            "o_preserved": len(viable) >= 2 and sum(row["dO"] < 0 for row in viable) >= 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=(1, 2), required=True)
    args = ap.parse_args()
    selfcheck()
    todo = [(s, a) for s in SEEDS for a in ARMS]
    if args.stage == 2:
        previous = json.loads((ROOT / "fcg_stage1.json").read_text())["contrasts"]
        todo = [(s, a) for s, a in todo if previous[f"{a}@{s}"]["class"] == "open"]
    out = {"schema": "teacher_v4_fcg_probe_v1", "stage": args.stage, "contrasts": {}, "dose_diagnostic": {}}
    for seed, arm in todo:
        pairs = [pair(seed, arm, r) for r in REPEATS[args.stage]]
        cls, med = classify(pairs, args.stage)
        raw = [p["raw_pp"] for p in pairs if p["valid"]]
        out["contrasts"][f"{arm}@{seed}"] = {"class": cls, "median_matched_pp": med,
            "median_raw_pp": float(np.median(raw)) if raw else None, "pairs": pairs}
    for seed in SEEDS:
        diagnostic = []
        for r in REPEATS[args.stage]:
            if not all((ROOT / f"seed{seed}" / f"{arm}_r{r}" / "replay" / "f2a_replay.json").exists()
                       for arm in ARMS):
                continue
            norm = arm_audit(seed, f"SPLIT-NORM-MATCHED_r{r}", ROOT)
            split = arm_audit(seed, f"SPLIT_r{r}", ROOT)
            valid = all(a["end_dR_matched"] is not None and not a["state"].startswith(("collapse", "unresolved"))
                        for a in (norm, split))
            diagnostic.append({"r": r, "valid": valid,
                               "matched_pp": 100 * (norm["end_dR_matched"] - split["end_dR_matched"]) if valid else None,
                               "raw_pp": 100 * (norm["end_dR"] - split["end_dR"]) if valid else None})
        out["dose_diagnostic"][str(seed)] = diagnostic
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / f"fcg_stage{args.stage}.json").write_text(json.dumps(out, indent=1))
    if args.stage == 1:
        (ROOT / "expand.txt").write_text("\n".join(k for k, v in out["contrasts"].items() if v["class"] == "open") + "\n")


def _check():
    def p(values):
        return [{"valid": x is not None, "matched_pp": x} for x in values]
    assert classify(p([12, 15, 3]), 1)[0] == "replicated attenuation"
    assert classify(p([-12, -15, -3]), 1)[0] == "replicated worsening"
    assert classify(p([1, -2, 4]), 1)[0] == "within practical band"
    assert classify(p([12, None, 4]), 1)[0] == "open"
    assert classify(p([12, 15, 3, 11, -1]), 2)[0] == "replicated attenuation"
    assert classify(p([12, 13, 14, None, None]), 2)[0] == "not resolved by registered budget"
    assert classify(p([10, 10, 10]), 1)[0] == "open"


_check()

if __name__ == "__main__":
    main()
