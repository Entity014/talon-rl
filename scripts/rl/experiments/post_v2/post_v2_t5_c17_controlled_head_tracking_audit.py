#!/usr/bin/env python3
"""Can a head fitted on a recent window predict the next batch?

Controlled comparison from one initial head: Adam for a few steps, or a ridge
closed form, each fitted on the last `window` batches and scored on the batch
after. Progress is measured in function space against a ridge fit over all six
batches, so a method that moves a long way without predicting better shows up.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rl.core.offline_audit import RUNS, OfflineAudit

SRC = "post_v2_t5_c13_conditioning-2026-09-23"
ORDER = ("T", "A", "O", "S")
BATCHES = 6
WINDOWS = (1, 2, 3, 4)
RIDGES = (0.1, 1.0)
ADAM_STEPS = (1, 5, 10, 25)
LR = 1e-4
HEADS = 4


def ev(y, p):
    y = np.asarray(y, float).reshape(-1)
    p = np.asarray(p, float).reshape(-1)
    return float(1 - np.var(y - p) / (np.var(y) + 1e-12))


def metrics(y, p):
    return {"ev": [ev(y[:, j], p[:, j]) for j in range(HEADS)],
            "mse": [float(np.mean((p[:, j] - y[:, j]) ** 2)) for j in range(HEADS)]}


def ridge_solution(f, y, l2):
    a = np.c_[f, np.ones(len(f))]
    i = np.eye(a.shape[1])
    i[-1, -1] = 0          # the intercept is not penalised
    sol = np.linalg.solve(a.T @ a + l2 * i, a.T @ y)
    return sol[:-1].T, sol[-1]


def cos(a, b):
    a, b = a.reshape(-1), b.reshape(-1)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def function_progress(f_ref, w0, b0, w, b, w_star, b_star):
    """Movement toward the stable fit, measured in predictions rather than weights."""
    p0 = f_ref @ w0.T + b0
    d = (f_ref @ w.T + b) - p0
    ds = (f_ref @ w_star.T + b_star) - p0
    return {"cosine": cos(d, ds),
            "norm_progress_ratio": float(np.linalg.norm(d) / (np.linalg.norm(ds) + 1e-12)),
            "projected_progress_ratio": float(
                (d.reshape(-1) @ ds.reshape(-1)) / (np.linalg.norm(ds) ** 2 + 1e-12))}


def adam_head(w0, b0, fw, yw, steps):
    w = torch.nn.Parameter(torch.tensor(w0.copy()))
    b = torch.nn.Parameter(torch.tensor(b0.copy()))
    opt = torch.optim.Adam([w, b], lr=LR)
    ft, yt = torch.tensor(fw), torch.tensor(yw)
    for _ in range(steps):
        loss = ((ft @ w.T + b - yt) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return w.detach().numpy(), b.detach().numpy()


class ControlledHeadTrackingAudit(OfflineAudit):
    """Window-fitted heads scored on the next batch, against a stable reference."""

    run = "post_v2_t5_c17_head_tracking-2026-09-23"
    report = "audit.json"
    schema = "t5_c17_controlled_head_tracking_v1"

    def specialist(self, lab):
        z = np.load(RUNS / SRC / f"{lab}_features_targets.npz")
        batches = [(z[f"F{k}"].astype(np.float32), z[f"Y{k}"].astype(np.float32))
                   for k in range(BATCHES)]
        w0 = z["head_weight"].astype(np.float32)
        b0 = z["head_bias"].astype(np.float32)

        f_all = np.concatenate([x[0] for x in batches], 0)
        y_all = np.concatenate([x[1] for x in batches], 0)
        w_star, b_star = ridge_solution(f_all.astype(np.float64), y_all.astype(np.float64), 1.0)
        # progress is scored on the whole feature support, not one batch's
        f_ref = f_all.astype(np.float64)

        out = {"reference": {"ridge_all6_lambda1": {"weight_norm": float(np.linalg.norm(w_star))}},
               "windows": []}
        for end in range(1, BATCHES):
            next_f, next_y = batches[end]
            for win in WINDOWS:
                sel = batches[max(0, end - win):end]
                fw = np.concatenate([x[0] for x in sel], 0)
                yw = np.concatenate([x[1] for x in sel], 0)
                row = {"end_batch": end - 1, "next_batch": end, "window": win,
                       "actual_window_size": len(sel), "methods": {}}
                row["methods"]["baseline"] = {
                    "window": metrics(yw, fw @ w0.T + b0),
                    "next": metrics(next_y, next_f @ w0.T + b0),
                    "progress": function_progress(f_ref, w0, b0, w0, b0, w_star, b_star)}
                for steps in ADAM_STEPS:
                    wn, bn = adam_head(w0, b0, fw, yw, steps)
                    row["methods"][f"adam_{steps}"] = {
                        "window": metrics(yw, fw @ wn.T + bn),
                        "next": metrics(next_y, next_f @ wn.T + bn),
                        "progress": function_progress(f_ref, w0, b0, wn, bn, w_star, b_star),
                        "weight_norm": float(np.linalg.norm(wn))}
                for l2 in RIDGES:
                    wr, br = ridge_solution(fw.astype(np.float64), yw.astype(np.float64), l2)
                    row["methods"][f"ridge_{l2}"] = {
                        "window": metrics(yw, fw @ wr.T + br),
                        "next": metrics(next_y, next_f @ wr.T + br),
                        "progress": function_progress(f_ref, w0, b0, wr, br, w_star, b_star),
                        "weight_norm": float(np.linalg.norm(wr))}
                out["windows"].append(row)
        return out

    def analyze(self):
        report = {"schema": self.schema,
                  "specialists": {lab: self.specialist(lab) for lab in ORDER}}
        report["aggregate"] = self.aggregate(report["specialists"])
        return report

    def aggregate(self, spec):
        methods = ["baseline"] + [f"adam_{s}" for s in ADAM_STEPS] + [f"ridge_{r}" for r in RIDGES]
        agg = {}
        for win in WINDOWS:
            agg[str(win)] = {}
            for method in methods:
                next_ev, window_ev, next_mse, prog, cosines = [], [], [], [], []
                for lab in ORDER:
                    for row in spec[lab]["windows"]:
                        if row["window"] != win:
                            continue
                        m = row["methods"][method]
                        next_ev += m["next"]["ev"]
                        window_ev += m["window"]["ev"]
                        next_mse += m["next"]["mse"]
                        prog.append(m["progress"]["projected_progress_ratio"])
                        cosines.append(m["progress"]["cosine"])
                agg[str(win)][method] = {
                    "window_ev_mean": float(np.mean(window_ev)),
                    "next_ev_mean": float(np.mean(next_ev)),
                    "next_ev_negative_fraction": float(np.mean(np.array(next_ev) < 0)),
                    "next_mse_mean": float(np.mean(next_mse)),
                    "projected_progress_mean": float(np.mean(prog)),
                    "function_cosine_mean": float(np.mean(cosines))}
        return agg

    def summarize(self, report):
        print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    ControlledHeadTrackingAudit.main()
