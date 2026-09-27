#!/usr/bin/env python3
"""V4-C2F aggregation: layers 1-3 of the objective specificity audit over the per-checkpoint specificity_audit.npz files, and the reading fixed in the contract.

Pure numpy; no simulator. Writes runs/teacher_v4_c2f_specificity_audit-2026-09-27/aggregate.json.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[6]
RUNS = REPO / "runs"
LABELS = ("T", "A", "O", "S")
PREFS = ("C", "T", "A", "O", "S")
PAIRS = list(itertools.combinations(range(4), 2))
PRIMARY = (73104, 73105, 73106, 73107, 73108)
BOOT = 2000


def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def offdiag(x, y):
    gx = x > np.median(x); gy = y > np.median(y)
    return float(np.mean(gx != gy))


def pc1(x, y):
    Z = np.stack([(x - x.mean()) / (x.std() + 1e-12), (y - y.mean()) / (y.std() + 1e-12)], 1)
    ev = np.linalg.eigvalsh(np.cov(Z.T))
    return float(ev.max() / ev.sum())


def pname(p): return LABELS[p[0]] + LABELS[p[1]]


def main():
    runs = []
    for s in PRIMARY + (73101, 73102, 73103):
        for fold, tag in (("G1-2", "g1_2"), ("G1-3", "g1_3")):
            f = RUNS / f"teacher_v4_c_{tag}_seed{s}-2026-09-27/specificity_audit.npz"
            runs.append((fold, s, s in PRIMARY, dict(np.load(f))))
    out = {"layers": {}}
    for scope, sel in (("primary", [r for r in runs if r[2]]), ("all", runs)):
        L1 = {}; L1w = {}; L2 = {}; L2pol = {}
        for p in PAIRS:
            a, b = p
            per_ck = [(spearman(np.concatenate([r[3][f"steps_{q}"] for q in PREFS])[:, a], np.concatenate([r[3][f"steps_{q}"] for q in PREFS])[:, b]),
                       offdiag(np.concatenate([r[3][f"steps_{q}"] for q in PREFS])[:, a], np.concatenate([r[3][f"steps_{q}"] for q in PREFS])[:, b])) for r in sel]
            L1[pname(p)] = {"spearman_median": float(np.median([x[0] for x in per_ck])), "offdiag_median": float(np.median([x[1] for x in per_ck]))}
            per_w = [(spearman(np.concatenate([r[3][f"win32_{q}"] for q in PREFS])[:, a], np.concatenate([r[3][f"win32_{q}"] for q in PREFS])[:, b]),
                      offdiag(np.concatenate([r[3][f"win32_{q}"] for q in PREFS])[:, a], np.concatenate([r[3][f"win32_{q}"] for q in PREFS])[:, b])) for r in sel]
            L1w[pname(p)] = {"spearman_median": float(np.median([x[0] for x in per_w])), "offdiag_median": float(np.median([x[1] for x in per_w]))}
            X = np.concatenate([r[3][f"steady_{q}"] for r in sel for q in PREFS])
            L2[pname(p)] = pc1(X[:, a], X[:, b])
            P = np.stack([r[3][f"steady_{q}"].mean(0) for r in sel for q in PREFS])
            L2pol[pname(p)] = pc1(P[:, a], P[:, b])
        out["layers"][scope] = {"L1_step": L1, "L1_win32": L1w, "L2_pc1_snapshots": L2, "L2_pc1_policies": L2pol}
    # layer 3: controllability matrix and specificity counts, per checkpoint
    rng = np.random.default_rng(0)
    L3 = []
    for fold, s, prim, d in runs:
        M = {q: d[f"steady_{q}"].mean(0).tolist() for q in PREFS}
        spec = {}
        for i in range(4):
            for j in range(4):
                if i == j: continue
                diff = d[f"steady_{LABELS[i]}"][:, i] - d[f"steady_{LABELS[j]}"][:, i]
                bm = diff[rng.integers(0, len(diff), (BOOT, len(diff)))].mean(1)
                lo, hi = np.percentile(bm, [2.5, 97.5])
                spec[f"{LABELS[i]}{LABELS[i]}-{LABELS[i]}{LABELS[j]}"] = {"mean": float(diff.mean()), "ci95": [float(lo), float(hi)],
                                                                          "sign": "pos" if lo > 0 else ("neg" if hi < 0 else "zero")}
        L3.append({"fold": fold, "seed": s, "primary": prim, "M": M, "specificity": spec})
    out["L3_per_checkpoint"] = L3
    counts = {}
    for scope, sel in (("primary", [x for x in L3 if x["primary"]]), ("all", L3)):
        keys = L3[0]["specificity"].keys()
        counts[scope] = {k: {sg: sum(x["specificity"][k]["sign"] == sg for x in sel) for sg in ("pos", "zero", "neg")} for k in keys}
    out["L3_counts"] = counts
    # reading (contract): A-O ranks among the six pairs on layers 1-2, and the A/O specificity block
    lay = out["layers"]["primary"]
    rank_hi = lambda d, k: sorted(d, key=lambda x: -d[x]).index(k) + 1  # noqa: E731
    rank_lo = lambda d, k: sorted(d, key=lambda x: d[x]).index(k) + 1   # noqa: E731
    reading = {}
    for pair in ("AO", "AS"):
        r = {"L1_step_abs_spearman_rank": rank_hi({k: abs(v["spearman_median"]) for k, v in lay["L1_step"].items()}, pair),
             "L1_win32_abs_spearman_rank": rank_hi({k: abs(v["spearman_median"]) for k, v in lay["L1_win32"].items()}, pair),
             "L1_step_offdiag_rank_low": rank_lo({k: v["offdiag_median"] for k, v in lay["L1_step"].items()}, pair),
             "L2_pc1_rank_high": rank_hi(lay["L2_pc1_snapshots"], pair)}
        c = counts["primary"]
        most_redundant = r["L1_step_abs_spearman_rank"] == 1 and r["L1_step_offdiag_rank_low"] == 1 and r["L2_pc1_rank_high"] == 1
        r["most_redundant_pair"] = most_redundant
        # the contract's rule, applied in both directions: x subsumed by y
        for x, y in ((pair[0], pair[1]), (pair[1], pair[0])):
            own = c[f"{x}{x}-{x}{y}"]; other = c[f"{y}{y}-{y}{x}"]
            own_not_pos = own["pos"] <= 5      # not positive in most of the 10 primary checkpoints
            other_pos = other["pos"] >= 6
            if most_redundant and own_not_pos and other_pos:
                rd = f"B: {x} subsumed by {y}"
            elif not most_redundant and own_not_pos:
                rd = f"A: {x} distinct from {y} but not realized"
            else:
                rd = "C: regime-dependent / layers disagree"
            r[f"{x}{x}-{x}{y}"] = own; r[f"{y}{y}-{y}{x}"] = other
            r[f"reading_{x}_vs_{y}"] = rd
        reading[pair] = r
    out["reading"] = reading
    dst = RUNS / "teacher_v4_c2f_specificity_audit-2026-09-27"; dst.mkdir(parents=True, exist_ok=True)
    (dst / "aggregate.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"L1_step": lay["L1_step"], "L1_win32": lay["L1_win32"], "L2": lay["L2_pc1_snapshots"], "L2_pol": lay["L2_pc1_policies"],
                      "L3_primary": counts["primary"], "reading": reading}, indent=1))


if __name__ == "__main__":
    main()
