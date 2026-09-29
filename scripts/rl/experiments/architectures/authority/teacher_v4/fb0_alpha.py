#!/usr/bin/env python3
"""FB-0 task-pressure calibration (docs/contracts/teacher_v4/teacher-v4-fb0-task-pressure-calibration-contract.md).

Offline on the FA bank: smallest alpha_T such that, at the R vertex, the O
vertex and the center of the R/O preference segment, the best locomoting
behavior beats the best non-locomoting one by more than m = 0.05 under
J = alpha T + (1 - alpha)(w_R R + w_O O), T3-B units.
"""
import json
from pathlib import Path

import numpy as np

from fa_geometry import REPO, T3B, bank

OUT = REPO / "runs/teacher_v4_fb-2026-09-29/fb0_alpha.json"
M = 0.05
POINTS = {"R-vertex": (1.0, 0.0), "O-vertex": (0.0, 1.0), "center": (0.5, 0.5)}


def main():
    rows = bank()
    loco = np.array([r["class"] in ("partial", "established") for r in rows])
    X = np.array([[r["raw"]["T"] / T3B["T"], r["raw"]["R"] / T3B["R"], r["raw"]["O"] / T3B["O"]] for r in rows])
    alphas = np.round(np.arange(0, 1.0001, 0.01), 2)

    def feasible(alpha, wr):
        J = alpha * X[:, 0] + (1 - alpha) * (wr * X[:, 1] + (1 - wr) * X[:, 2])
        return J[loco].max() - J[~loco].max() > M

    first = lambda pred: next((float(a) for a in alphas if pred(a)), None)  # noqa: E731
    per_point = {k: first(lambda a, w=w: feasible(a, w[0])) for k, w in POINTS.items()}
    alpha_T = first(lambda a: all(feasible(a, w[0]) for w in POINTS.values()))
    segment = first(lambda a: all(feasible(a, wr) for wr in np.arange(0, 1.0001, 0.01)))
    out = {"schema": "teacher_v4_fb0_alpha_v1", "margin": M, "n_bank": len(rows), "n_locomoting": int(loco.sum()),
           "alpha_T": alpha_T, "lambda_T": None if alpha_T in (None, 1.0) else alpha_T / (1 - alpha_T),
           "per_point_min_alpha": per_point, "whole_segment_min_alpha": segment}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
