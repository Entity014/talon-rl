#!/usr/bin/env python3
"""F5 actor-update geometry screen, A0 vs A2 (docs/contracts/teacher_v4/teacher-v4-f5-update-geometry-contract.md).

Offline. Per run: gait phenotype per checkpoint from the F2-A replay
(N no translation, H costly translation, L low-cost translation), persistent
L at C (primary) and the T+ phenotype, plus optimizer traces; then the
pre-declared per-arm reading.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from f3_screen import D_MIN, O_MIN, persistent

ARMS = {"A0": 0.01, "A2": 0.02}
SEEDS = {"screen": (75101, 75102, 75103, 75104, 75105), "confirm": (75201, 75202, 75203, 75204, 75205)}
WINDOWS = ((1, 100), (101, 300), (301, 600))


def phenotype(p):
    if p["tl"] < 0.40:
        return "N"
    return "L" if p["R"][1] >= D_MIN and p["R"][2] >= O_MIN else "H"


def run_summary(d):
    ck = json.load(open(d / "replay" / "f2a_replay.json"))["checkpoints"]; its = sorted(int(k) for k in ck)
    out = {"stopped": (d / "stopped.pt").exists(), "last_checkpoint": its[-1]}
    for c in ("C", "T+"):
        ph = [phenotype(ck[str(u)][c]) for u in its]
        out[c] = {"phenotype_by_checkpoint": dict(zip(its, ph)), "final": ph[-1],
                  "t_L_persistent": persistent([x == "L" for x in ph], its),
                  "t_translate_persistent": persistent([x != "N" for x in ph], its),
                  "final_R": ck[str(its[-1])][c]["R"], "final_tl": ck[str(its[-1])][c]["tl"]}
    rows = [json.loads(l) for l in open(d / "metrics.jsonl")]
    it = np.array([r["iteration"] for r in rows])
    out["optimizer"] = {f"{a}-{b}": {k: float(np.mean([f(r) for r, i in zip(rows, it) if a <= i <= b])) if ((it >= a) & (it <= b)).any() else None
                                     for k, f in (("kl", lambda r: r["kl"]), ("clip_frac", lambda r: r["clip_frac"]), ("lr", lambda r: r["lr"]),
                                                  ("log_std", lambda r: r["log_std"]["mean"]), ("termination", lambda r: r["termination_fraction"]),
                                                  ("preference_authority", lambda r: r["preference_authority"]))}
                        for a, b in WINDOWS}
    out["L_C"] = out["C"]["t_L_persistent"] is not None and not out["stopped"]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=tuple(SEEDS), required=True)
    p.add_argument("--root", required=True, help="runs/teacher_v4_f5-2026-09-28")
    a = p.parse_args()
    root = Path(a.root)
    res = {arm: {s: run_summary(root / f"{arm}_seed{s}") for s in SEEDS[a.stage]} for arm in ARMS}
    k0 = sum(r["L_C"] for r in res["A0"].values()); k2 = sum(r["L_C"] for r in res["A2"].values())
    reading = ("A2 screen-positive" if k2 >= 2 and k2 >= k0 + 2 else "update-geometry hypothesis not supported" if k2 <= k0 else "inconclusive")

    def counts(arm, c):
        return {ph: sum(r[c]["final"] == ph for r in res[arm].values()) for ph in "NHL"}

    out = {"schema": "teacher_v4_f5_screen_v1", "stage": a.stage, "L_C": {"A0": k0, "A2": k2}, "reading": reading,
           "final_phenotype": {arm: {c: counts(arm, c) for c in ("C", "T+")} for arm in ARMS},
           "runs": {arm: {str(s): v for s, v in r.items()} for arm, r in res.items()}}
    json.dump(out, open(root / f"f5_{a.stage}.json", "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("L_C", "reading", "final_phenotype")}, indent=1))


if __name__ == "__main__":
    main()
