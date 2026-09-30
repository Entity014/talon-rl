#!/usr/bin/env python3
"""FC-F staged paired matched-dR screen; rule in the FC-F contract."""
import argparse
import json
from pathlib import Path

import numpy as np

from fcc_probe import arm_audit, selfcheck

ROOT = Path(__file__).resolve().parents[6] / "runs/teacher_v4_fc-2026-09-30/fcf"
SEEDS = (79101, 79103)
ARMS = ("NO-PREF-PATH", "NO-BASES", "SHARED-FEATURES-ONLY", "PREF-ONLY")
REPEATS = {1: (6, 7, 8), 2: (6, 7, 8, 9, 10)}


def classify(pairs, stage):
    e = [p["matched_pp"] for p in pairs if p["valid"]]
    med = float(np.median(e)) if e else None
    need = 3 if stage == 1 else 4
    if len(e) >= need and sum(x > 0 for x in e) >= need and med > 10:
        return "replicated attenuation", med
    if len(e) >= need and sum(x < 0 for x in e) >= need and med < -10:
        return "replicated worsening", med
    if len(e) >= need and sum(abs(x) < 10 for x in e) >= need and abs(med) < 5:
        return "within practical band", med
    return ("open" if stage == 1 else "not resolved by registered budget"), med


def pair(seed, arm, repeat):
    a = arm_audit(seed, f"FULL_r{repeat}", ROOT)
    b = arm_audit(seed, f"{arm}_r{repeat}", ROOT)
    valid = all(v["end_dR_matched"] is not None and not v["state"].startswith(("collapse", "unresolved")) for v in (a, b))
    return {"r": repeat, "valid": valid, "states": [a["state"], b["state"]],
            "matched_pp": 100 * (a["end_dR_matched"] - b["end_dR_matched"]) if valid else None,
            "raw_pp": 100 * (a["end_dR"] - b["end_dR"]) if valid else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=(1, 2), required=True)
    args = ap.parse_args()
    selfcheck()
    todo = [(s, a) for s in SEEDS for a in ARMS]
    if args.stage == 2:
        previous = json.loads((ROOT / "fcf_stage1.json").read_text())["contrasts"]
        todo = [(s, a) for s, a in todo if previous[f"{a}@{s}"]["class"] == "open"]
    out = {"schema": "teacher_v4_fcf_probe_v1", "stage": args.stage, "contrasts": {}}
    for seed, arm in todo:
        pairs = [pair(seed, arm, r) for r in REPEATS[args.stage]]
        cls, med = classify(pairs, args.stage)
        raw = [p["raw_pp"] for p in pairs if p["valid"]]
        out["contrasts"][f"{arm}@{seed}"] = {"class": cls, "median_matched_pp": med,
            "median_raw_pp": float(np.median(raw)) if raw else None, "pairs": pairs}
        print(seed, arm, cls, med, flush=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / f"fcf_stage{args.stage}.json").write_text(json.dumps(out, indent=1))
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
    assert classify(p([1, -2, 4, 3, 11]), 2)[0] == "within practical band"
    assert classify(p([12, 13, 14, None, None]), 2)[0] == "not resolved by registered budget"
    assert classify(p([10, 10, 10]), 1)[0] == "open"  # strict threshold
    assert classify(p([12, None, -4, 4, 2]), 2)[0] == "not resolved by registered budget"


_check()

if __name__ == "__main__":
    main()
