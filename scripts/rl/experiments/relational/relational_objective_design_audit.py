#!/usr/bin/env python3
"""Does a relation-based objective track semantic correctness better than J?

Offline only, no optimizer steps. Compares the change in the heavy objective
against the change in the pure relational term, then a family of candidates
that weight the two, at three values of eta. A candidate is authorised only if
the pure relation beats J outright and every eta passes every criterion —
including that a gain is rarely driven by the objective centre degrading.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rl.core.offline_audit import REPO, RUNS, OfflineAudit
from rl.experiments.shared.rank_stats import corr, spearman

SRC = "relational_persistence_heldout-2026-09-24/heldout_report.json"
CONTRACT = "docs/relational-objective-design-audit-contract.md"
AXES = ("T", "A", "O", "S")
ETAS = (0.25, 0.50, 1.00)
TOL = 1e-10


def metric(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    mask = (np.abs(x) > TOL) & (np.abs(y) > TOL)
    return {"n": len(x), "spearman": spearman(x, y), "pearson": corr(x, y),
            "sign_agreement": (float(np.mean(np.sign(x[mask]) == np.sign(y[mask])))
                               if mask.any() else float("nan")),
            "sign_n": int(mask.sum())}


def frac(part, whole, default=float("nan")):
    return float(len(part) / len(whole)) if whole else default


class RelationalObjectiveDesignAudit(OfflineAudit):
    """Whether a relational objective earns its place over the heavy objective."""

    run = "relational_objective_design_audit-2026-09-24"
    report = "relational_objective_design_report.json"
    schema = "relational_objective_design_audit_v1"

    def samples(self):
        states = json.loads((RUNS / SRC).read_text())["states"]
        out = []
        for seed, arr in states.items():
            for i in range(len(arr) - 1):
                a, b = arr[i], arr[i + 1]
                for axis in AXES:
                    x, y = a["axes"][axis], b["axes"][axis]
                    jh0, jh1 = float(x["J_heavy"]), float(y["J_heavy"])
                    r0, r1 = float(x["obj_adv_mean"]), float(y["obj_adv_mean"])
                    out.append({
                        "seed": int(seed), "from": a["label"], "to": b["label"], "axis": axis,
                        "dJH": jh1 - jh0, "dJC": (jh1 - r1) - (jh0 - r0), "dR": r1 - r0,
                        "d_obj_correct": float(y["objective_correct_fraction"] - x["objective_correct_fraction"]),
                        "d_phys_correct": float(y["physical_correct_fraction"] - x["physical_correct_fraction"]),
                        "d_semantic_score": float(y["semantic_score"] - x["semantic_score"]),
                        "pass_from": bool(x["PASS"]), "pass_to": bool(y["PASS"])})
        return out

    def eta_block(self, samples, eta, base_obj):
        vals = []
        for s in samples:
            heavy = (1 + eta) * s["dJH"]
            center = -eta * s["dJC"]
            q = dict(s)
            q.update({"dCand": heavy + center, "heavy_contribution": heavy,
                      "center_contribution": center,
                      "guardrail_ok": bool(s["dJH"] >= -TOL),
                      "candidate_positive": bool(heavy + center > TOL)})
            vals.append(q)

        cand_obj = metric([s["dCand"] for s in vals], [s["d_obj_correct"] for s in vals])
        cand_sem = metric([s["dCand"] for s in vals], [s["d_semantic_score"] for s in vals])
        pos = [s for s in vals if s["dCand"] > TOL]
        center_false = [s for s in pos if s["dJH"] <= TOL]
        center_dom = [s for s in pos if s["center_contribution"] > max(s["heavy_contribution"], 0.0)]
        sem_false_obj = [s for s in pos if s["d_obj_correct"] < -TOL]
        sem_false_sem = [s for s in pos if s["d_semantic_score"] < -TOL]
        pf = [s for s in vals if s["pass_from"] and not s["pass_to"]]
        pf_approved = [s for s in pf if s["dCand"] > TOL]
        pf_guard_approved = [s for s in pf if s["dCand"] > TOL and s["guardrail_ok"]]
        guard = [s for s in vals if s["guardrail_ok"]]
        guard_pos = [s for s in guard if s["dCand"] > TOL]
        guard_obj = (metric([s["dCand"] for s in guard], [s["d_obj_correct"] for s in guard])
                     if len(guard) >= 2 else None)
        guard_sem = (metric([s["dCand"] for s in guard], [s["d_semantic_score"] for s in guard])
                     if len(guard) >= 2 else None)

        per_seed, signs = {}, []
        for sd in sorted({s["seed"] for s in vals}):
            ss = [s for s in vals if s["seed"] == sd]
            rr = spearman([s["dCand"] for s in ss], [s["d_obj_correct"] for s in ss])
            per_seed[str(sd)] = rr
            if np.isfinite(rr):
                signs.append(np.sign(rr))
        overall_sign = np.sign(cand_obj["spearman"]) if np.isfinite(cand_obj["spearman"]) else 0
        sign_cons = bool(len(signs) == 3 and all(x == overall_sign for x in signs))
        keep_frac = frac(guard_pos, pos, 0.0)

        c = {"spearman_gain_ge_0p10": cand_obj["spearman"] >= base_obj["spearman"] + .10,
             "semantic_false_gain_obj_le_0p25": frac(sem_false_obj, pos, 1.0) <= .25,
             "center_degradation_false_gain_le_0p10": frac(center_false, pos, 1.0) <= .10,
             "guardrailed_pass_fail_false_approval_le_0p20": frac(pf_guard_approved, pf, 1.0) <= .20,
             "guardrail_retains_ge_0p60_positive": keep_frac >= .60,
             "seed_sign_consistency_3_of_3": sign_cons}
        block = {
            "eta": eta, "candidate_obj_metric": cand_obj, "candidate_sem_metric": cand_sem,
            "guardrail_obj_metric": guard_obj, "guardrail_sem_metric": guard_sem,
            "candidate_positive_n": len(pos), "guardrail_admissible_n": len(guard),
            "guardrail_positive_n": len(guard_pos),
            "guardrail_positive_retention_fraction": keep_frac,
            "center_degradation_false_gain_fraction": frac(center_false, pos),
            "center_dominated_positive_fraction": frac(center_dom, pos),
            "semantic_false_gain_obj_fraction": frac(sem_false_obj, pos),
            "semantic_false_gain_score_fraction": frac(sem_false_sem, pos),
            "pass_to_fail_n": len(pf),
            "pass_to_fail_false_approval_fraction": frac(pf_approved, pf),
            "guardrailed_pass_to_fail_false_approval_fraction": frac(pf_guard_approved, pf),
            "per_seed_spearman": per_seed, "seed_sign_consistent": sign_cons, "criteria": c,
            "center_false_gain_events": list(center_false)}
        return block, c

    def analyze(self):
        samples = self.samples()
        base_obj = metric([s["dJH"] for s in samples], [s["d_obj_correct"] for s in samples])
        base_sem = metric([s["dJH"] for s in samples], [s["d_semantic_score"] for s in samples])
        rel_obj = metric([s["dR"] for s in samples], [s["d_obj_correct"] for s in samples])
        rel_sem = metric([s["dR"] for s in samples], [s["d_semantic_score"] for s in samples])

        per_eta, criteria_by_eta = {}, {}
        for eta in ETAS:
            per_eta[str(eta)], criteria_by_eta[str(eta)] = self.eta_block(samples, eta, base_obj)

        rel_confirm = rel_obj["spearman"] >= base_obj["spearman"] + .15
        all_eta = all(all(c.values()) for c in criteria_by_eta.values())
        criteria = {"pure_relation_beats_J_by_0p15": rel_confirm, "all_eta_pass": all_eta}
        authorized = bool(rel_confirm and all_eta)
        status = ("AUTHORIZED" if authorized
                  else "RELATION VALID BUT CANDIDATE PATHOLOGICAL" if rel_confirm
                  else "NO BENEFIT")
        return {"schema": self.schema, "status": status, "offline_only": True,
                "optimizer_steps": 0, "sample_count": len(samples), "eta_values": list(ETAS),
                "baseline_J_obj_metric": base_obj, "baseline_J_sem_metric": base_sem,
                "pure_relation_obj_metric": rel_obj, "pure_relation_sem_metric": rel_sem,
                "per_eta": per_eta, "criteria_by_eta": criteria_by_eta, "criteria": criteria,
                "decision": {"objective_design_branch_authorized": authorized},
                "samples": samples}

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
        keep = ["candidate_obj_metric", "candidate_sem_metric",
                "guardrail_positive_retention_fraction", "center_degradation_false_gain_fraction",
                "center_dominated_positive_fraction", "semantic_false_gain_obj_fraction",
                "semantic_false_gain_score_fraction", "pass_to_fail_false_approval_fraction",
                "guardrailed_pass_to_fail_false_approval_fraction", "per_seed_spearman",
                "criteria"]
        print(json.dumps({"status": rep["status"], "criteria": rep["criteria"],
                          "baseline_J_obj_metric": rep["baseline_J_obj_metric"],
                          "pure_relation_obj_metric": rep["pure_relation_obj_metric"],
                          "per_eta": {k: {z: v[z] for z in keep}
                                      for k, v in rep["per_eta"].items()}}, indent=2))


if __name__ == "__main__":
    RelationalObjectiveDesignAudit.main()
