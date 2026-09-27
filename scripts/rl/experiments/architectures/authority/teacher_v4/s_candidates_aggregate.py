#!/usr/bin/env python3
"""V4-C2S aggregation and the frozen selection rule over the 10 primary s_candidates_audit.npz files.

Writes runs/teacher_v4_c2s_smoothness_audit-2026-09-27/aggregate.json.
Also reports, descriptively, the same measures within each preference
(pooled over checkpoints but not over preferences), since pooling across
preferences mixes in between-policy gait-intensity differences.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from specificity_aggregate import offdiag, pc1, spearman

REPO = Path(__file__).resolve().parents[6]
RUNS = REPO / "runs"
COLS = ("T", "A", "O", "S0", "S1", "S2", "S3")
PREFS = ("C", "T", "A", "O", "S")
PRIMARY = (73104, 73105, 73106, 73107, 73108)
CAND = ("S0", "S1", "S2", "S3")
AO_STEP, AO_WIN = 0.16, 0.23           # V4-C2F A-O medians (eligibility bound)
TIE_ORDER = ("S3", "S1", "S2")


def measures(runs, x, y, prefs=PREFS):
    i, j = COLS.index(x), COLS.index(y)
    st = [spearman(*(np.concatenate([r[f"steps_{p}"] for p in prefs])[:, k] for k in (i, j))) for r in runs]
    wi = [spearman(*(np.concatenate([r[f"win32_{p}"] for p in prefs])[:, k] for k in (i, j))) for r in runs]
    od = [offdiag(*(np.concatenate([r[f"steps_{p}"] for p in prefs])[:, k] for k in (i, j))) for r in runs]
    X = np.concatenate([r[f"steady_{p}"] for r in runs for p in prefs])
    return {"rho_step": float(np.median(st)), "rho_win32": float(np.median(wi)), "offdiag_step": float(np.median(od)),
            "pc1_snap": pc1(X[:, i], X[:, j])}


def main():
    runs = [dict(np.load(RUNS / f"teacher_v4_c_{tag}_seed{s}-2026-09-27/s_candidates_audit.npz"))
            for s in PRIMARY for tag in ("g1_2", "g1_3")]
    res = {c: {ref: measures(runs, ref, c) for ref in ("A", "T", "O")} for c in CAND}
    eligible = {c: all(abs(res[c][ref]["rho_step"]) <= AO_STEP and abs(res[c][ref]["rho_win32"]) <= AO_WIN for ref in ("T", "O")) for c in CAND}
    keys = (("rho_step", True), ("rho_win32", True), ("offdiag_step", False), ("pc1_snap", True))  # True: lower |x| is better
    el = [c for c in CAND if eligible[c]]
    ranks = {}
    for c in el:
        rs = []
        for k, lower in keys:
            vals = {cc: (abs(res[cc]["A"][k]) if lower else -res[cc]["A"][k]) for cc in el}
            rs.append(sorted(vals, key=vals.get).index(c) + 1)
        ranks[c] = float(np.mean(rs))
    selected = None
    if ranks:
        best = min(ranks.values())
        tied = [c for c in ranks if ranks[c] == best]
        cand = next((c for c in TIE_ORDER if c in tied), tied[0])
        beats = all((abs(res[cand]["A"][k]) < abs(res["S0"]["A"][k])) if lower else (res[cand]["A"][k] > res["S0"]["A"][k]) for k, lower in keys) if cand != "S0" else False
        selected = cand if beats else None
    within = {p: {c: measures(runs, "A", c, prefs=(p,)) for c in CAND} for p in PREFS}
    out = {"measures": res, "eligible": eligible, "mean_rank_vs_A": ranks, "selected": selected, "within_preference_vs_A": within}
    dst = RUNS / "teacher_v4_c2s_smoothness_audit-2026-09-27"; dst.mkdir(parents=True, exist_ok=True)
    (dst / "aggregate.json").write_text(json.dumps(out, indent=2) + "\n")
    for c in CAND:
        print(c, "vs A", {k: round(v, 3) for k, v in res[c]["A"].items()}, "| vs T rho", round(res[c]["T"]["rho_step"], 3), round(res[c]["T"]["rho_win32"], 3),
              "| vs O rho", round(res[c]["O"]["rho_step"], 3), round(res[c]["O"]["rho_win32"], 3), "| eligible", eligible[c])
    print("mean rank vs A:", ranks, "SELECTED:", selected)
    for p in PREFS:
        print("within", p, {c: round(within[p][c]["rho_step"], 2) for c in CAND})


if __name__ == "__main__":
    main()
