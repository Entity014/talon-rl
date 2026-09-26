#!/usr/bin/env python3
"""Is online head learning moving toward the solution a stable fit would pick?

Freezes two reference solutions on the u25 body and matched targets — ridge and
rank-32 PCR — then measures how far the online head has travelled from u0 in
their direction, overall and per head. A negative cosine means it is moving
away from what the data supports.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rl.core.offline_audit import RUNS, OfflineAudit

C14 = "post_v2_t5_c14_representation_drift-2026-09-23"
ORDER = ("T", "A", "O", "S")
SNAPS = (0, 10, 25)
RIDGE = 1.0
PCR_RANK = 32
HEADS = 4


def cos(a, b):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def ev(y, p):
    y = np.asarray(y, float).reshape(-1)
    p = np.asarray(p, float).reshape(-1)
    return float(1 - np.var(y - p) / (np.var(y) + 1e-12))


def ridge_solution(f, y, l2):
    a = np.c_[f, np.ones(len(f))]
    i = np.eye(a.shape[1])
    i[-1, -1] = 0          # the intercept is not penalised
    sol = np.linalg.solve(a.T @ a + l2 * i, a.T @ y)
    return sol[:-1].T, sol[-1]


def pcr_solution(f, y, rank):
    mu = f.mean(0, keepdims=True)
    x = f - mu
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    r = min(rank, len(s))
    vr = vt[:r].T
    a = np.c_[x @ vr, np.ones(len(x))]
    sol = np.linalg.lstsq(a, y, rcond=None)[0]
    w_red = sol[:-1].T
    return w_red @ vr.T, sol[-1] - w_red @ (mu.reshape(-1) @ vr), s


def displacement_metrics(w0, b0, wt, bt, wstar, bstar):
    """How the online step from u0 compares with the step to the stable fit."""
    d_on = np.r_[(wt - w0).reshape(-1), (bt - b0).reshape(-1)]
    d_st = np.r_[(wstar - w0).reshape(-1), (bstar - b0).reshape(-1)]
    n_on, n_st = np.linalg.norm(d_on), np.linalg.norm(d_st)
    rho_par = float((d_on @ d_st) / (d_st @ d_st + 1e-12))
    orth = np.linalg.norm(d_on - rho_par * d_st)
    return {"online_norm": float(n_on), "target_norm": float(n_st),
            "cosine": cos(d_on, d_st),
            "norm_progress_ratio": float(n_on / (n_st + 1e-12)),
            "projected_progress_ratio": rho_par,
            "orthogonal_fraction_of_online": float(orth / (n_on + 1e-12))}


class HeadProgressAudit(OfflineAudit):
    """Online head progress against frozen ridge and PCR references."""

    run = "post_v2_t5_c16_head_progress-2026-09-23"
    report = "audit.json"
    schema = "t5_c16_head_progress_v1"

    def specialist(self, lab):
        z = np.load(RUNS / C14 / f"{lab}_matched.npz")
        y = z["Y"].astype(np.float64)
        f = {s: z[f"F{s}"].astype(np.float64) for s in SNAPS}
        w = {s: z[f"W{s}"].astype(np.float64) for s in SNAPS}
        b = {s: z[f"b{s}"].astype(np.float64) for s in SNAPS}

        # references frozen on the u25 body and matched target, robust classes only
        wr, br = ridge_solution(f[25], y, RIDGE)
        wp, bp, _ = pcr_solution(f[25], y, PCR_RANK)

        out = {"ridge_ref": {}, "pcr_ref": {}, "actual_ev": {}}
        for t in (10, 25):
            out["ridge_ref"][str(t)] = displacement_metrics(w[0], b[0], w[t], b[t], wr, br)
            out["pcr_ref"][str(t)] = displacement_metrics(w[0], b[0], w[t], b[t], wp, bp)
        for s in SNAPS:
            pred = f[s] @ w[s].T + b[s]
            out["actual_ev"][str(s)] = [ev(y[:, j], pred[:, j]) for j in range(HEADS)]

        per_head = {}
        for j in range(HEADS):
            sl = slice(j, j + 1)
            per_head[str(j)] = {
                "ridge": {str(t): displacement_metrics(w[0][sl], b[0][sl], w[t][sl], b[t][sl],
                                                       wr[sl], br[sl]) for t in (10, 25)},
                "pcr": {str(t): displacement_metrics(w[0][sl], b[0][sl], w[t][sl], b[t][sl],
                                                     wp[sl], bp[sl]) for t in (10, 25)}}
        out["per_head"] = per_head
        out["reference_ev"] = {
            "ridge_on_u25": [ev(y[:, j], (f[25] @ wr.T + br)[:, j]) for j in range(HEADS)],
            "pcr32_on_u25": [ev(y[:, j], (f[25] @ wp.T + bp)[:, j]) for j in range(HEADS)]}
        return out

    def analyze(self):
        report = {"schema": self.schema,
                  "stable_refs": {"ridge_lambda": RIDGE, "pcr_rank": PCR_RANK},
                  "specialists": {lab: self.specialist(lab) for lab in ORDER}}
        report["aggregate"] = self.aggregate(report["specialists"])
        return report

    def aggregate(self, spec):
        mean = lambda vals, k: float(np.mean([x[k] for x in vals]))  # noqa: E731
        agg = {"ridge": {}, "pcr": {}}
        for refkey, outkey in (("ridge_ref", "ridge"), ("pcr_ref", "pcr")):
            for t in ("10", "25"):
                vals = [spec[lab][refkey][t] for lab in ORDER]
                cosines = np.array([x["cosine"] for x in vals])
                agg[outkey][t] = {
                    "cosine_mean": mean(vals, "cosine"),
                    "cosine_median": float(np.median(cosines)),
                    "cosine_negative_fraction": float(np.mean(cosines < 0)),
                    "norm_progress_ratio_mean": mean(vals, "norm_progress_ratio"),
                    "projected_progress_ratio_mean": mean(vals, "projected_progress_ratio"),
                    "projected_progress_ratio_median": float(
                        np.median([x["projected_progress_ratio"] for x in vals])),
                    "orthogonal_fraction_mean": mean(vals, "orthogonal_fraction_of_online")}
        agg["per_head"] = {}
        for refname in ("ridge", "pcr"):
            agg["per_head"][refname] = {}
            for j in range(HEADS):
                for t in ("10", "25"):
                    vals = [spec[lab]["per_head"][str(j)][refname][t] for lab in ORDER]
                    agg["per_head"][refname].setdefault(str(j), {})[t] = {
                        "cosine_mean": mean(vals, "cosine"),
                        "projected_progress_ratio_mean": mean(vals, "projected_progress_ratio"),
                        "norm_progress_ratio_mean": mean(vals, "norm_progress_ratio")}
        return agg

    def summarize(self, report):
        print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    HeadProgressAudit.main()
