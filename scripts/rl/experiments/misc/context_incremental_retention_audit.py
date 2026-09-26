#!/usr/bin/env python3
"""Do context features predict gate retention beyond the mean margins alone?

Read-only. Fits leave-one-seed-out logistic regressions on the two mean
margins, then on those plus seven context features, and asks whether the
larger model earns its keep on every criterion at once. A nonparametric check
inside each margin stratum backs the parametric one up, so a gain cannot come
from the fit alone.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rl.core.offline_audit import REPO, RUNS, OfflineAudit

SRC = "semantic_gate_factorization_audit-2026-09-24/semantic_gate_factorization_report.json"
CONTRACT = "docs/context-incremental-retention-contract.md"
SEEDS = (980001, 981001, 982001)
AXES = ("T", "A", "O", "S")
BASE_FEATS = ["obj_mean_margin", "phys_mean_margin"]
CTX_FEATS = ["obj_phase_pos_frac", "phys_phase_pos_frac",
             "obj_worst_suite_phase_margin", "phys_worst_suite_phase_margin",
             "obj_std_margin", "phys_std_margin", "obj_phys_sign_agreement_fraction"]
SUITE_PHASE_CELLS = 12.0
ROBUST_CTX = .75


def safe_auc(y, p):
    y, p = np.asarray(y, int), np.asarray(p, float)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = sum(1.0 if p[i] > p[j] else 0.5 if p[i] == p[j] else 0.0 for i in pos for j in neg)
    return float(wins / (len(pos) * len(neg)))


def safe_ap(y, p):
    y, p = np.asarray(y, int), np.asarray(p, float)
    if y.sum() == 0:
        return float("nan")
    ys = y[np.argsort(-p, kind="mergesort")]
    prec = np.cumsum(ys) / np.arange(1, len(ys) + 1)
    return float(np.sum(prec * ys) / y.sum())


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def fit_logistic_l2(x, y, c=1.0, steps=4000, lr=0.05):
    x, y = np.asarray(x, float), np.asarray(y, float)
    w, b, lam = np.zeros(x.shape[1]), 0.0, 1.0 / c
    for _ in range(steps):
        err = sigmoid(x @ w + b) - y
        w -= lr * (x.T @ err / len(y) + lam * w / len(y))
        b -= lr * float(err.mean())
    return w, b


def fit_predict(train, test, features):
    xtr = np.array([[x[f] for f in features] for x in train], float)
    xte = np.array([[x[f] for f in features] for x in test], float)
    ytr = np.array([x["retained"] for x in train], int)
    mu, sd = xtr.mean(0), xtr.std(0)
    sd[sd < 1e-12] = 1.0
    xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
    if len(set(ytr)) < 2:
        return np.full(len(test), float(ytr.mean()))
    w, b = fit_logistic_l2(xtr, ytr, c=1.0)
    return sigmoid(xte @ w + b)


def metrics(rows, p):
    y = np.array([r["retained"] for r in rows], int)
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return {"n": len(rows), "positive_retained": int(y.sum()), "forgotten": int((1 - y).sum()),
            "roc_auc": safe_auc(y, p), "average_precision": safe_ap(y, p),
            "brier": float(np.mean((p - y) ** 2)),
            "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))}


class ContextIncrementalRetentionAudit(OfflineAudit):
    """Incremental value of context features for predicting gate retention."""

    run = "context_incremental_retention_audit-2026-09-24"
    report = "context_incremental_retention_report.json"
    schema = "context_incremental_retention_audit_v1"

    def rows(self):
        states = json.loads((RUNS / SRC).read_text())["states"]
        out = []
        for s in SEEDS:
            arr = states[str(s)]
            for i in range(len(arr) - 1):
                for axis in AXES:
                    src, dst = arr[i]["axes"][axis], arr[i + 1]["axes"][axis]
                    if not src["PASS"]:
                        continue    # only a state that passed can be forgotten
                    out.append({
                        "seed": s, "from": arr[i]["label"], "to": arr[i + 1]["label"],
                        "axis": axis, "retained": int(dst["PASS"]),
                        "obj_mean_margin": float(src["obj_mean_margin"]),
                        "phys_mean_margin": float(src["phys_mean_margin"]),
                        "obj_phase_pos_frac": 1.0 - float(src["obj_negative_suite_phase_cells"]) / SUITE_PHASE_CELLS,
                        "phys_phase_pos_frac": 1.0 - float(src["phys_negative_suite_phase_cells"]) / SUITE_PHASE_CELLS,
                        "obj_worst_suite_phase_margin": float(src["obj_worst_suite_phase_margin"]),
                        "phys_worst_suite_phase_margin": float(src["phys_worst_suite_phase_margin"]),
                        "obj_std_margin": float(src["obj_std_margin"]),
                        "phys_std_margin": float(src["phys_std_margin"]),
                        "obj_phys_sign_agreement_fraction": float(src["obj_phys_sign_agreement_fraction"])})
        return out

    def strata(self, rows):
        """Within each seed's upper/lower margin half, does context still separate?"""
        comp = []
        for s in SEEDS:
            rr = [r for r in rows if r["seed"] == s]
            vals = np.array([[r["obj_mean_margin"], r["phys_mean_margin"]] for r in rr], float)
            mu, sd = vals.mean(0), vals.std(0)
            sd[sd < 1e-12] = 1
            b = ((vals - mu) / sd).sum(1)
            med = float(np.median(b))
            for r, bv in zip(rr, b):
                q = dict(r)
                q["stratum"] = "upper" if bv >= med else "lower"
                q["ctx_robust"] = bool(r["obj_phase_pos_frac"] >= ROBUST_CTX
                                       and r["phys_phase_pos_frac"] >= ROBUST_CTX)
                comp.append(q)
        out = {}
        for st in ("lower", "upper"):
            ss = [x for x in comp if x["stratum"] == st]
            a = [x for x in ss if x["ctx_robust"]]
            b = [x for x in ss if not x["ctx_robust"]]
            ra = float(np.mean([x["retained"] for x in a])) if a else float("nan")
            rb = float(np.mean([x["retained"] for x in b])) if b else float("nan")
            out[st] = {"n": len(ss), "robust_n": len(a), "nonrobust_n": len(b),
                       "robust_retention": ra, "nonrobust_retention": rb,
                       "difference": ra - rb if np.isfinite(ra) and np.isfinite(rb) else float("nan")}
        return out

    def analyze(self):
        rows = self.rows()
        pred_base, pred_ctx, ordered, per_seed = [], [], [], {}
        for held in SEEDS:
            tr = [r for r in rows if r["seed"] != held]
            te = [r for r in rows if r["seed"] == held]
            pb = fit_predict(tr, te, BASE_FEATS)
            pc = fit_predict(tr, te, BASE_FEATS + CTX_FEATS)
            mb, mc = metrics(te, pb), metrics(te, pc)
            per_seed[str(held)] = {
                "base": mb, "base_ctx": mc,
                "auc_delta": (mc["roc_auc"] - mb["roc_auc"]
                              if np.isfinite(mc["roc_auc"]) and np.isfinite(mb["roc_auc"])
                              else float("nan")),
                "brier_delta": mc["brier"] - mb["brier"]}
            ordered.extend(te)
            pred_base.extend(pb.tolist())
            pred_ctx.extend(pc.tolist())
        pooled_base, pooled_ctx = metrics(ordered, pred_base), metrics(ordered, pred_ctx)
        strata = self.strata(rows)

        n = len(rows)
        forget = sum(1 - r["retained"] for r in rows)
        under = n < 20 or forget < 8
        seed_nonworse = True
        for v in per_seed.values():
            if np.isfinite(v["auc_delta"]):
                seed_nonworse &= v["auc_delta"] >= -1e-12
            else:
                seed_nonworse &= v["brier_delta"] <= 1e-12
        diffs = [strata[x]["difference"] for x in ("lower", "upper")]
        nonparam = (all(np.isfinite(x) for x in diffs)
                    and max(diffs) >= .15 and min(diffs) >= -1e-12)
        criteria = {
            "sample_not_underpowered": not under,
            "auc_gain_ge_0p10": pooled_ctx["roc_auc"] >= pooled_base["roc_auc"] + .10,
            "ap_gain_ge_0p10": pooled_ctx["average_precision"] >= pooled_base["average_precision"] + .10,
            "brier_improvement_ge_0p02": pooled_ctx["brier"] <= pooled_base["brier"] - .02,
            "logloss_improves": pooled_ctx["log_loss"] < pooled_base["log_loss"],
            "nonworse_all_3_seeds": bool(seed_nonworse),
            "nonparametric_context_gain": bool(nonparam)}
        if under:
            status = "UNDERPOWERED / INCONCLUSIVE"
        elif all(criteria.values()):
            status = "INCREMENTAL CONTEXT VALUE SUPPORTED"
        else:
            any_gain = any([criteria["auc_gain_ge_0p10"], criteria["ap_gain_ge_0p10"],
                            criteria["brier_improvement_ge_0p02"],
                            criteria["nonparametric_context_gain"]])
            status = "INCONCLUSIVE" if any_gain else "NO INCREMENTAL VALUE"
        return {"schema": self.schema, "status": status, "read_only": True,
                "pass_origin_n": n, "forgetting_n": forget, "retained_n": n - forget,
                "pooled": {"base": pooled_base, "base_ctx": pooled_ctx,
                           "auc_gain": pooled_ctx["roc_auc"] - pooled_base["roc_auc"],
                           "ap_gain": pooled_ctx["average_precision"] - pooled_base["average_precision"],
                           "brier_improvement": pooled_base["brier"] - pooled_ctx["brier"],
                           "logloss_improvement": pooled_base["log_loss"] - pooled_ctx["log_loss"]},
                "per_seed": per_seed, "nonparametric_strata": strata, "criteria": criteria,
                "decision": {"context_intervention_design_authorized":
                             status == "INCREMENTAL CONTEXT VALUE SUPPORTED"},
                "rows": rows}

    def execute(self):
        rep = super().execute()
        self.write({"status": "FROZEN_BY_HASH", "decision": rep["status"],
                    "report_sha256": self.sha(self.out / self.report),
                    "script_sha256": self.sha(Path(__file__).resolve()),
                    "contract_sha256": self.sha(REPO / CONTRACT),
                    "source_sha256": self.sha(RUNS / SRC)},
                   "PROVENANCE_MANIFEST.json")
        return rep

    def summarize(self, rep):
        print(json.dumps({k: rep[k] for k in
                          ["status", "pass_origin_n", "forgetting_n", "retained_n", "pooled",
                           "per_seed", "nonparametric_strata", "criteria"]}, indent=2))


if __name__ == "__main__":
    ContextIncrementalRetentionAudit.main()
