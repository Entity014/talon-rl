#!/usr/bin/env python3
"""Principal-component regression conditioning of the C13 critic features.

Fits each feature block's targets in a truncated PCA basis, then applies that
fit to the next block, to see how far a rank-limited solution carries.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rl.core.offline_audit import OfflineAudit

ORDER = ("T", "A", "O", "S")
BLOCKS = 6
RANKS = (16, 32, 48, 64, 96, 128)


def ev(y, p):
    """Explained variance of a prediction against its target."""
    y = np.asarray(y, float).reshape(-1)
    p = np.asarray(p, float).reshape(-1)
    return float(1 - np.var(y - p) / (np.var(y) + 1e-12))


def metrics(y, p):
    return {"ev": [ev(y[:, j], p[:, j]) for j in range(4)]}


class PCRAudit(OfflineAudit):
    """Principal-component regression conditioning of the C13 critic features."""

    run = "post_v2_t5_c13_conditioning-2026-09-23"
    report = "pcr_audit.json"

    def analyze(self):
        out = {"ranks": RANKS, "specialists": {}}
        for lab in ORDER:
            z = self.load(f"{lab}_features_targets.npz")
            blocks = [(z[f"F{k}"].astype(np.float64), z[f"Y{k}"].astype(np.float64))
                      for k in range(BLOCKS)]
            labout = {}
            for rank in RANKS:
                rows = []
                for k in range(BLOCKS - 1):
                    f, y = blocks[k]
                    fn, yn = blocks[k + 1]
                    mu = f.mean(0)
                    x, xn = f - mu, fn - mu
                    _, s, vt = np.linalg.svd(x, full_matrices=False)
                    r = min(rank, len(s))
                    vr = vt[:r].T
                    a = np.c_[x @ vr, np.ones(len(x))]
                    an = np.c_[xn @ vr, np.ones(len(xn))]
                    sol = np.linalg.lstsq(a, y, rcond=None)[0]
                    rows.append({"pair": f"{k}->{k+1}",
                                 "self": metrics(y, a @ sol),
                                 "prior_on_next": metrics(yn, an @ sol),
                                 "retained_singular_energy": float(np.sum(s[:r] ** 2) / np.sum(s ** 2))})
                labout[str(rank)] = rows
            out["specialists"][lab] = labout
        out["aggregate"] = self.aggregate(out)
        return out

    def aggregate(self, out):
        agg = {}
        for rank in RANKS:
            se, ne, energy = [], [], []
            for lab in ORDER:
                for row in out["specialists"][lab][str(rank)]:
                    se += row["self"]["ev"]
                    ne += row["prior_on_next"]["ev"]
                    energy.append(row["retained_singular_energy"])
            agg[str(rank)] = {"self_ev_mean": float(np.mean(se)),
                              "next_ev_mean": float(np.mean(ne)),
                              "next_ev_negative_fraction": float(np.mean(np.array(ne) < 0)),
                              "retained_singular_energy_mean": float(np.mean(energy))}
        return agg

    def summarize(self, report):
        print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    PCRAudit.main()
