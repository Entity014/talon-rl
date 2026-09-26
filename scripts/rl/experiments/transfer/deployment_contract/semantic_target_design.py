#!/usr/bin/env python3
"""Which scalar target tracks the semantic score best, and can it be gamed?

Offline only, no optimizer steps. Scores four candidate targets on held-out
transitions: the heavy objective alone, the rescaled objective advantage, their
mean, and a soft minimum of the two. A candidate has to correlate well, approve
few of the transitions that lost the gate, and survive a guardrail that drops
gains the heavy objective disagrees with.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rl.core.diagnostics.offline_audit import REPO, RUNS, OfflineAudit
from rl.experiments.common.utilities.rank_stats import corr, spearman

DISC = "trajectory_information_attribution_audit-2026-09-24/trajectory_information_attribution_report.json"
HOLD = "relational_persistence_heldout-2026-09-24/heldout_report.json"
CONTRACT = "docs/contracts/transfer/phase1-semantic-target-design-contract.md"
AXES = ("T", "A", "O", "S")
TARGETS = ("B0", "B1", "B2", "C")
TOL = 1e-10





def softmin2(a, b):
    """A smooth minimum, so neither channel can be traded away entirely."""
    m = min(-a, -b)
    lse = m + math.log(math.exp(-a - m) + math.exp(-b - m))
    return -(lse - math.log(2.0))


def metrics(samples, key):
    x = [s["delta"][key] for s in samples]
    y = [s["d_sem"] for s in samples]
    mask = [abs(a) > TOL and abs(b) > TOL for a, b in zip(x, y)]
    sg = (float(np.mean([np.sign(a) == np.sign(b) for a, b, m in zip(x, y, mask) if m]))
          if any(mask) else float("nan"))
    perseed = {str(sd): spearman([s["delta"][key] for s in samples if s["seed"] == sd],
                              [s["d_sem"] for s in samples if s["seed"] == sd])
               for sd in sorted({s["seed"] for s in samples})}
    peraxis = {a: spearman([s["delta"][key] for s in samples if s["axis"] == a],
                        [s["d_sem"] for s in samples if s["axis"] == a]) for a in AXES}
    pf = [s for s in samples if s["pass_from"] and not s["pass_to"]]
    fp = [s for s in samples if not s["pass_from"] and s["pass_to"]]
    pos = [s for s in samples if s["delta"][key] > TOL]
    return {"spearman": spearman(x, y), "pearson": corr(x, y), "sign_agreement": sg,
            "per_seed_spearman": perseed, "per_axis_spearman": peraxis,
            "pass_to_fail_n": len(pf),
            "pass_to_fail_false_approval_fraction":
                float(np.mean([s["delta"][key] > TOL for s in pf])) if pf else float("nan"),
            "fail_to_pass_n": len(fp),
            "fail_to_pass_true_approval_fraction":
                float(np.mean([s["delta"][key] > TOL for s in fp])) if fp else float("nan"),
            "positive_n": len(pos),
            "positive_semantic_deterioration_fraction":
                float(np.mean([s["d_sem"] < -TOL for s in pos])) if pos else float("nan")}


class SemanticTargetDesignAudit(OfflineAudit):
    """Stage A: choose a scalar semantic target from held-out transitions."""

    run = "phase1_semantic_target_design_audit-2026-09-24"
    report = "phase1_semantic_target_design_report.json"
    schema = "phase1_semantic_target_design_audit_v1"

    def scales(self, disc):
        """Per-axis median |advantage|, so objective and physical share a scale."""
        out = {}
        for a in AXES:
            om, pm = [], []
            for arr in disc["states"].values():
                for st in arr:
                    q = st["axes"][a]
                    om.append(abs(float(q["aggregate.obj_adv_mean"])))
                    pm.append(abs(float(q["aggregate.phys_adv_mean"])))
            out[a] = {"obj": float(np.median(om) + 1e-8), "phys": float(np.median(pm) + 1e-8)}
        return out

    def samples(self, hold, scales):
        out = []
        for seed, arr in hold["states"].items():
            for i in range(len(arr) - 1):
                for a in AXES:
                    sc = scales[a]
                    x, y = arr[i]["axes"][a], arr[i + 1]["axes"][a]

                    def scores(q, sc=sc):
                        zo = float(q["obj_adv_mean"]) / sc["obj"]
                        zp = float(q["phys_adv_mean"]) / sc["phys"]
                        return {"B0": float(q["J_heavy"]), "B1": zo, "B2": 0.5 * (zo + zp),
                                "C": softmin2(zo, zp), "zobj": zo, "zphys": zp}

                    sx, sy = scores(x), scores(y)
                    jcx = float(x["J_heavy"]) - float(x["obj_adv_mean"])
                    jcy = float(y["J_heavy"]) - float(y["obj_adv_mean"])
                    out.append({
                        "seed": int(seed), "from": arr[i]["label"], "to": arr[i + 1]["label"],
                        "axis": a, "pass_from": bool(x["PASS"]), "pass_to": bool(y["PASS"]),
                        "d_sem": float(y["semantic_score"] - x["semantic_score"]),
                        "d_obj_corr": float(y["objective_correct_fraction"] - x["objective_correct_fraction"]),
                        "d_phys_corr": float(y["physical_correct_fraction"] - x["physical_correct_fraction"]),
                        "dJH": float(y["J_heavy"] - x["J_heavy"]), "dJC_obj": jcy - jcx,
                        "delta": {k: sy[k] - sx[k] for k in TARGETS},
                        "score_from": sx, "score_to": sy})
        return out

    def analyze(self):
        disc = json.loads((RUNS / DISC).read_text())
        hold = json.loads((RUNS / HOLD).read_text())
        scales = self.scales(disc)
        samples = self.samples(hold, scales)
        res = {k: metrics(samples, k) for k in TARGETS}

        # can a candidate gain come from the objective centre rather than the heavy axis?
        pos = [s for s in samples if s["delta"]["C"] > TOL]
        guard = [s for s in pos if s["dJH"] >= -TOL]
        pf = [s for s in samples if s["pass_from"] and not s["pass_to"]]
        pfg = [s for s in pf if s["dJH"] >= -TOL]
        center_false = [s for s in pos if s["dJH"] < -TOL]
        center_driven = [s for s in pos if s["dJC_obj"] < 0 and -s["dJC_obj"] > max(s["dJH"], 0)]
        guard_pf_false = float(np.mean([s["delta"]["C"] > TOL for s in pfg])) if pfg else 0.0
        guard_keep = float(len(guard) / len(pos)) if pos else 0.0

        seed_sign = all(np.isfinite(v) and v > 0 for v in res["C"]["per_seed_spearman"].values())
        fp_n = res["C"]["fail_to_pass_n"]
        fp_ok = (res["C"]["fail_to_pass_true_approval_fraction"] >= .60) if fp_n >= 4 else True
        criteria = {
            "spearman_ge_0p65": res["C"]["spearman"] >= .65,
            "beats_B0_by_0p20": res["C"]["spearman"] >= res["B0"]["spearman"] + .20,
            "beats_B1_by_0p05": res["C"]["spearman"] >= res["B1"]["spearman"] + .05,
            "pass_fail_false_approval_le_0p15": res["C"]["pass_to_fail_false_approval_fraction"] <= .15,
            "pass_fail_at_least_0p10_better_than_B2":
                res["C"]["pass_to_fail_false_approval_fraction"]
                <= res["B2"]["pass_to_fail_false_approval_fraction"] - .10,
            "fail_pass_true_approval_ge_0p60_if_powered": fp_ok,
            "positive_semantic_deterioration_le_0p15":
                res["C"]["positive_semantic_deterioration_fraction"] <= .15,
            "positive_correlation_all_3_seeds": seed_sign,
            "guardrail_no_worse_pass_fail_and_retains_ge_0p60":
                guard_pf_false <= res["C"]["pass_to_fail_false_approval_fraction"] + 1e-12
                and guard_keep >= .60}
        passed = all(criteria.values())
        return {"schema": self.schema, "status": "STAGE A PASS" if passed else "STAGE A FAIL",
                "offline_only": True, "optimizer_steps": 0,
                "scales_from_discovery": scales, "sample_count": len(samples), "results": res,
                "candidate_pathology": {
                    "candidate_positive_n": len(pos),
                    "objective_heavy_guardrail_keep_fraction": guard_keep,
                    "objective_center_degradation_false_gain_fraction":
                        float(len(center_false) / len(pos)) if pos else float("nan"),
                    "objective_center_dominated_fraction":
                        float(len(center_driven) / len(pos)) if pos else float("nan"),
                    "guardrailed_pass_to_fail_false_approval_fraction": guard_pf_false},
                "criteria": criteria,
                "decision": {"stage_b_physical_decomposition_authorized": passed,
                             "training_pilot_authorized": False},
                "samples": samples}

    def execute(self):
        rep = super().execute()
        self.write({"status": "FROZEN_BY_HASH", "decision": rep["status"],
                    "report_sha256": self.sha(self.out / self.report),
                    "script_sha256": self.sha(Path(__file__).resolve()),
                    "contract_sha256": self.sha(REPO / CONTRACT),
                    "discovery_sha256": self.sha(RUNS / DISC),
                    "heldout_sha256": self.sha(RUNS / HOLD)},
                   "PROVENANCE_MANIFEST.json")
        return rep

    def summarize(self, rep):
        print(json.dumps({"status": rep["status"], "scales": rep["scales_from_discovery"],
                          "results": rep["results"],
                          "candidate_pathology": rep["candidate_pathology"],
                          "criteria": rep["criteria"]}, indent=2))


if __name__ == "__main__":
    SemanticTargetDesignAudit.main()
