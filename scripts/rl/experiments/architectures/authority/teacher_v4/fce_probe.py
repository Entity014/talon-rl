#!/usr/bin/env python3
"""FC-E paired replication of the task-pressure contrasts
(docs/contracts/teacher_v4/teacher-v4-fce-paired-replication-contract.md).

Offline. For each contrast (seed, arm1, arm2), repeat r runs both arms from
the same u0 with the same branch seed (common random numbers); the paired
effect is endpoint Delta_R(arm1) - Delta_R(arm2) in percentage points,
matched primary, raw alongside. Sequential rule: stage 1 = repeats 1-3,
stage 2 adds 4-5 only for contrasts stage 1 leaves open. Writes the list of
contrasts to expand (stage 1) or the final classes (stage 2).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from fcc_probe import arm_audit, selfcheck

ROOT = Path(__file__).resolve().parents[6] / "runs/teacher_v4_fc-2026-09-30/fce"
CONTRASTS = {"task@79101": (79101, "A", "D1", "primary"), "task@79103": (79103, "A", "D1", "primary"),
             "region@79101": (79101, "D2p", "D2", "primary"), "task@79102": (79102, "A", "D1", "secondary")}
STAGE_REPEATS = {1: (1, 2, 3), 2: (1, 2, 3, 4, 5)}
EFFECT_PP = 10.0


def pair(s, a1, a2, r):
    x, y = arm_audit(s, f"{a1}_r{r}", ROOT), arm_audit(s, f"{a2}_r{r}", ROOT)
    ok = all(v["end_dR_matched"] is not None and not v["state"].startswith(("collapse", "unresolved")) for v in (x, y))
    return {"r": r, "valid": ok, "states": [x["state"], y["state"]],
            "matched_pp": 100 * (x["end_dR_matched"] - y["end_dR_matched"]) if ok else None,
            "raw_pp": 100 * (x["end_dR"] - y["end_dR"]) if ok else None,
            "arm_matched": [x["end_dR_matched"], y["end_dR_matched"]], "arm_raw": [x["end_dR"], y["end_dR"]]}


def classify(pairs, stage):
    e = [p["matched_pp"] for p in pairs if p["valid"]]
    med = float(np.median(e)) if e else None
    if stage == 1:
        if len(e) == 3 and all(x > 0 for x in e) and med > EFFECT_PP:
            return "replicated", med
        if len(e) == 3 and all(x < 0 for x in e) and med < -EFFECT_PP:
            return "reversed", med
        return "open (expand to 5)", med
    pos, neg = sum(x > 0 for x in e), sum(x < 0 for x in e)
    if len(e) >= 4 and pos >= 4 and med > EFFECT_PP:
        return "replicated", med
    if len(e) >= 4 and neg >= 4 and med < -EFFECT_PP:
        return "reversed", med
    return "not replicated", med


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", type=int, choices=(1, 2), required=True); a = ap.parse_args()
    selfcheck()
    if a.stage == 1:
        todo = CONTRASTS
    else:
        s1 = json.load(open(ROOT / "fce_stage1.json"))["contrasts"]
        todo = {k: v for k, v in CONTRASTS.items() if s1[k]["class"].startswith("open")}
    out = {"schema": "teacher_v4_fce_probe_v1", "stage": a.stage, "contrasts": {}}
    for name, (s, a1, a2, role) in todo.items():
        pairs = [pair(s, a1, a2, r) for r in STAGE_REPEATS[a.stage]]
        cls, med = classify(pairs, a.stage)
        e = np.array([p["matched_pp"] for p in pairs if p["valid"]])
        out["contrasts"][name] = {"seed": s, "arms": [a1, a2], "role": role, "class": cls, "median_matched_pp": med,
                                  "median_raw_pp": float(np.median([p["raw_pp"] for p in pairs if p["valid"]])) if len(e) else None,
                                  "sd_matched_pp": float(e.std(ddof=1)) if len(e) > 1 else None, "pairs": pairs}
    json.dump(out, open(ROOT / f"fce_stage{a.stage}.json", "w"), indent=1)
    if a.stage == 1:
        (ROOT / "expand.txt").write_text("\n".join(k for k, v in out["contrasts"].items() if v["class"].startswith("open")) + "\n")
    for k, v in out["contrasts"].items():
        print(f"STAGE{a.stage}", k, v["role"], v["class"], "median matched", None if v["median_matched_pp"] is None else round(v["median_matched_pp"], 1), "pp",
              "raw", None if v["median_raw_pp"] is None else round(v["median_raw_pp"], 1),
              "| pairs", [(p["r"], None if p["matched_pp"] is None else round(p["matched_pp"], 1), None if p["raw_pp"] is None else round(p["raw_pp"], 1)) for p in v["pairs"]])


def _check():  # fails if the sequential rule changes meaning
    P = lambda v: [{"valid": True, "matched_pp": x} for x in v]  # noqa: E731
    assert classify(P([12, 15, 11]), 1)[0] == "replicated"
    assert classify(P([12, 15, -1]), 1)[0].startswith("open")
    assert classify(P([12, 8, 9]), 1)[0].startswith("open")
    assert classify(P([-12, -15, -11]), 1)[0] == "reversed"
    assert classify(P([12, 15, -1, 11, 20]), 2)[0] == "replicated"
    assert classify(P([12, 15, -1, -3, 20]), 2)[0] == "not replicated"


_check()

if __name__ == "__main__":
    main()
