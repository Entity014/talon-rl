#!/usr/bin/env python3
"""Are the semantic-gate pass-to-fail events robust to how the gate is scored?

Rescales each axis by the discovery run's median margin, checks that the
rescaled gate reproduces the recorded PASS exactly, then for every recorded
pass-to-fail event measures boundary clearance, a threshold sweep and a
resampling of the four suites.
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rl.core.offline_audit import REPO, OfflineAudit

AXES = ("T", "A", "O", "S")
FACTORIZATION = ("semantic_gate_factorization_audit-2026-09-24/"
                 "semantic_gate_factorization_report.json")
DISCOVERY = ("trajectory_information_attribution_audit-2026-09-24/"
             "trajectory_information_attribution_report.json")
CONTRACT = "docs/semantic-gate-robustness-audit-contract.md"
TAUS = (-0.25, 0.0, 0.25)


def gate_margin(vals):
    """The second-smallest margin: the gate needs three of four suites."""
    return float(np.sort(np.asarray(vals, float))[1])


def pass_at(obj, phys, tau=0.0):
    return bool(np.sum(np.asarray(obj) > tau) >= 3 and np.sum(np.asarray(phys) > tau) >= 3)


class SemanticGateRobustnessAudit(OfflineAudit):
    """Robustness of the semantic-gate pass-to-fail events to gate scoring."""

    run = "semantic_gate_robustness_audit-2026-09-24"
    report = "semantic_gate_robustness_report.json"
    schema = "semantic_gate_robustness_audit_v1"

    def scales(self, disc):
        """Per-axis median |advantage|, used to put both channels on one scale."""
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

    def rescaled_states(self, fac, scales):
        """Every state's rescaled margins, plus any disagreement with recorded PASS."""
        states, mismatch = {}, []
        for sd, arr in fac["states"].items():
            rows = []
            for st in arr:
                axes = {}
                for a in AXES:
                    q = st["axes"][a]
                    zo = [float(x["obj_margin"]) / scales[a]["obj"] for x in q["suite_factors"]]
                    zp = [float(x["phys_margin"]) / scales[a]["phys"] for x in q["suite_factors"]]
                    go, gp = gate_margin(zo), gate_margin(zp)
                    gs = min(go, gp)
                    if (gs > 0) != bool(q["PASS"]):
                        mismatch.append((sd, st["label"], a, gs, q["PASS"]))
                    axes[a] = {"z_obj": zo, "z_phys": zp, "G_obj": go, "G_phys": gp,
                               "G_sem": gs, "PASS": bool(q["PASS"])}
                rows.append({"label": st["label"], "axes": axes})
            states[sd] = rows
        return states, mismatch

    def event(self, e, states):
        sd, a = str(e["seed"]), e["axis"]
        arr = states[sd]
        s = next(x for x in arr if x["label"] == e["from"])["axes"][a]
        t = next(x for x in arr if x["label"] == e["to"])["axes"][a]
        clearance = min(s["G_sem"], -t["G_sem"])

        sweep = {str(tau): {"source_pass": pass_at(s["z_obj"], s["z_phys"], tau),
                            "target_pass": pass_at(t["z_obj"], t["z_phys"], tau)}
                 for tau in TAUS}
        sweep_rob = all(v["source_pass"] and not v["target_pass"] for v in sweep.values())

        flips = srcpass = tgtpass = 0
        for idxs in itertools.product(range(4), repeat=4):
            ps = pass_at([s["z_obj"][k] for k in idxs], [s["z_phys"][k] for k in idxs], 0)
            pt = pass_at([t["z_obj"][k] for k in idxs], [t["z_phys"][k] for k in idxs], 0)
            srcpass += ps
            tgtpass += pt
            flips += (ps and not pt)
        pflip, psrc, ptgt = flips / 256, srcpass / 256, tgtpass / 256

        if clearance < 0.10 or pflip < 0.50 or not sweep_rob:
            cls = "A"
        elif clearance >= 0.25 and sweep_rob and pflip >= 0.75:
            cls = "B"
        else:
            cls = "MIXED"
        return {"seed": e["seed"], "from": e["from"], "to": e["to"], "axis": a,
                "G_source": s["G_sem"], "G_target": t["G_sem"], "clearance": clearance,
                "delta_G": t["G_sem"] - s["G_sem"], "sweep": sweep,
                "threshold_sweep_robust_flip": sweep_rob,
                "bootstrap_source_pass_prob": psrc, "bootstrap_target_pass_prob": ptgt,
                "bootstrap_pass_to_fail_prob": pflip, "class": cls,
                "target_limiting_channel": "objective" if t["G_obj"] <= t["G_phys"] else "physical"}

    def analyze(self):
        fac = self.load(Path(FACTORIZATION).name, str(Path(FACTORIZATION).parent))
        disc = self.load(Path(DISCOVERY).name, str(Path(DISCOVERY).parent))
        scales = self.scales(disc)
        states, mismatch = self.rescaled_states(fac, scales)
        events = [self.event(e, states) for e in fac["pass_to_fail_events"]]

        n = len(events)
        counts = {k: sum(e["class"] == k for e in events) for k in ("A", "B", "MIXED")}
        medc = float(np.median([e["clearance"] for e in events]))
        medb = float(np.median([e["bootstrap_pass_to_fail_prob"] for e in events]))
        bseeds = {e["seed"] for e in events if e["class"] == "B"}
        robust = (counts["B"] / n >= .60 and counts["A"] / n <= .25
                  and medc >= .25 and medb >= .75 and len(bseeds) == 3)
        sens = counts["A"] / n >= .50 or medc < .10 or medb < .50
        status = ("FORGETTING ROBUST" if robust
                  else "EVALUATOR SENSITIVITY MATERIAL" if sens else "MIXED / INCONCLUSIVE")
        return {"schema": self.schema, "status": status, "read_only": True, "scale": scales,
                "equivalence_mismatches": mismatch, "event_count": n, "class_counts": counts,
                "class_fractions": {k: counts[k] / n for k in counts},
                "median_boundary_clearance": medc,
                "median_bootstrap_pass_to_fail_probability": medb,
                "class_B_seeds": sorted(bseeds), "events": events,
                "decision": {"training_method_authorized": False,
                             "evaluation_contract_change_authorized": False}}

    def execute(self):
        rep = super().execute()
        self.write({"status": "FROZEN_BY_HASH", "decision": rep["status"],
                    "report_sha256": self.sha(self.out / self.report),
                    "script_sha256": self.sha(Path(__file__).resolve()),
                    "contract_sha256": self.sha(REPO / CONTRACT),
                    "factorization_sha256": self.sha(self.dir.parent / FACTORIZATION),
                    "discovery_sha256": self.sha(self.dir.parent / DISCOVERY)},
                   "PROVENANCE_MANIFEST.json")
        return rep

    def summarize(self, rep):
        print(json.dumps({k: rep[k] for k in
                          ("status",)} | {"equivalence_mismatches": len(rep["equivalence_mismatches"])}
                         | {k: rep[k] for k in ("class_counts", "class_fractions",
                                                "median_boundary_clearance",
                                                "median_bootstrap_pass_to_fail_probability",
                                                "events")}, indent=2))


if __name__ == "__main__":
    SemanticGateRobustnessAudit.main()
