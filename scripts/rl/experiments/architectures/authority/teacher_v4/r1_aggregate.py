#!/usr/bin/env python3
"""V4-C2S-R1 aggregation (docs/contracts/teacher_v4/teacher-v4-c2s-r1-action-jerk-contract.md).

Treatment: runs/teacher_v4_c2s1_*; control: the six V4-C G1 runs. Uses the
s_candidates_audit.npz of each (columns T, A, O, S0, S1, S2, S3), with S := S1.
Layers 1-2 over the pairs among T, A, O, S1 for each group; layer 3 (SS-SA,
AA-AS with S1-heavy) for the treatment; the frozen outcome rule; the paired
per-seed change in rho(A, S1). Writes runs/teacher_v4_c2s_r1-2026-09-27/aggregate.json.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from specificity_aggregate import offdiag, pc1, spearman

REPO = Path(__file__).resolve().parents[6]
RUNS = REPO / "runs"
COLS = ("T", "A", "O", "S0", "S1", "S2", "S3")
USE = {"T": 0, "A": 1, "O": 2, "S": 4}           # S := S1
PREFS = ("C", "T", "A", "O", "S")
SEEDS = (73101, 73102, 73103)
BOOT = 2000


def load(prefix):
    return [(tag, s, dict(np.load(RUNS / f"{prefix}_{tag}_seed{s}-2026-09-27/s_candidates_audit.npz")))
            for tag in ("g1_2", "g1_3") for s in SEEDS]


def layers(runs):
    out = {}
    for a, b in itertools.combinations("TAOS", 2):
        i, j = USE[a], USE[b]
        cat = lambda r, key: np.concatenate([r[f"{key}_{p}"] for p in PREFS])  # noqa: E731
        rs = [spearman(cat(r, "steps")[:, i], cat(r, "steps")[:, j]) for _, _, r in runs]
        rw = [spearman(cat(r, "win32")[:, i], cat(r, "win32")[:, j]) for _, _, r in runs]
        od = [offdiag(cat(r, "steps")[:, i], cat(r, "steps")[:, j]) for _, _, r in runs]
        X = np.concatenate([r[f"steady_{p}"] for _, _, r in runs for p in PREFS])
        out[a + b] = {"rho_step": float(np.median(rs)), "rho_win32": float(np.median(rw)), "offdiag_step": float(np.median(od)),
                      "pc1_snap": pc1(X[:, i], X[:, j]), "rho_step_per_run": rs}
    return out


def spec_signs(runs, x, y, rng):
    """CI sign of S_x(x+) - S_x(y+) per run (steady means per snapshot)."""
    k = USE[x]; out = []
    for _, _, r in runs:
        d = r[f"steady_{x}"][:, k] - r[f"steady_{y}"][:, k]
        bm = d[rng.integers(0, len(d), (BOOT, len(d)))].mean(1)
        lo, hi = np.percentile(bm, [2.5, 97.5])
        out.append("pos" if lo > 0 else ("neg" if hi < 0 else "zero"))
    return out


def main():
    rng = np.random.default_rng(0)
    T = load("teacher_v4_c2s1"); C = load("teacher_v4_c")
    LT, LC = layers(T), layers(C)
    ss_sa = spec_signs(T, "S", "A", rng); aa_as = spec_signs(T, "A", "S", rng)
    ranks = {}
    for m, higher_is_redundant in (("rho_step", True), ("offdiag_step", False), ("pc1_snap", True)):
        vals = {p: (abs(v[m]) if m.startswith("rho") else v[m]) for p, v in LT.items()}
        order = sorted(vals, key=lambda p: -vals[p] if higher_is_redundant else vals[p])
        ranks[m] = order.index("AS") + 1
    as_most = all(r == 1 for r in ranks.values())
    as_not_most = all(r != 1 for r in ranks.values())
    elig = all(abs(LT[p]["rho_step"]) <= 0.16 and abs(LT[p]["rho_win32"]) <= 0.23 for p in ("TS", "OS"))
    n_pos = ss_sa.count("pos")
    if as_not_most and elig and n_pos >= 4:
        outcome = "separates"
    elif as_most and n_pos <= 3:
        outcome = "still_redundant"
    else:
        outcome = "mixed"
    paired = [{"run": f"{tt}_{s}", "rho_A_S1_treat": float(LT["AS"]["rho_step_per_run"][k]), "rho_A_S1_ctrl": float(LC["AS"]["rho_step_per_run"][k])}
              for k, (tt, s, _) in enumerate(T)]
    out = {"treatment_layers": LT, "control_layers": LC, "AS_redundancy_ranks_treatment": ranks, "eligibility_vs_T_O": elig,
           "SS-SA_signs": ss_sa, "AA-AS_signs": aa_as, "outcome": outcome, "paired_rho_A_S1": paired}
    dst = RUNS / "teacher_v4_c2s_r1-2026-09-27"; dst.mkdir(parents=True, exist_ok=True)
    (dst / "aggregate.json").write_text(json.dumps(out, indent=2) + "\n")
    for name, L in (("TREAT", LT), ("CTRL", LC)):
        print(name, {p: (round(v["rho_step"], 2), round(v["rho_win32"], 2), round(v["offdiag_step"], 2), round(v["pc1_snap"], 2)) for p, v in L.items()})
    print("AS ranks", ranks, "elig", elig, "SS-SA", ss_sa, "AA-AS", aa_as, "OUTCOME", outcome)
    print("paired", [(p["run"], round(p["rho_A_S1_ctrl"], 2), round(p["rho_A_S1_treat"], 2)) for p in paired])


if __name__ == "__main__":
    main()
