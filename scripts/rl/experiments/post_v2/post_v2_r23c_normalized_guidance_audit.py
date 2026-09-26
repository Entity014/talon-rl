#!/usr/bin/env python3
"""How much normalised guidance is needed to orient the pairwise semantics?

Read-only algebra over the R2.3-A coefficient pressures: for each budget rho,
add a guidance vector of norm rho * ||PPO pressure|| and check whether the
three preference-pair differences then point the way they should. Reports the
smallest rho that works at every snapshot, and what the combination costs in
cosine against PPO's own direction. No optimizer or model update.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rl.core.offline_audit import REPO, OfflineAudit

RHO = (0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
EPS = 1e-12
TARGETS = {"PB": np.array([1., -1.]), "PE": np.array([1., 1.]), "BE": np.array([0., 1.])}
PAIRS = (("P", "B", "PB"), ("P", "E", "PE"), ("B", "E", "BE"))


def sign(x, tol=1e-12):
    return 0 if abs(x) <= tol else (1 if x > 0 else -1)


def pair_ok(v, t):
    """A zero in the target means that component is unconstrained."""
    if t[0] and sign(v[0]) != int(t[0]):
        return False
    if t[1] and sign(v[1]) != int(t[1]):
        return False
    return True


def cosine(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + EPS))


class NormalizedGuidanceAudit(OfflineAudit):
    """Smallest guidance budget that orients every preference pair."""

    run = "post_v2_r23c_normalized_guidance-2026-09-23"
    report = "audit.json"
    schema = "post_v2_r23c_normalized_guidance_audit_v1"

    def __init__(self, out=None, source=None):
        super().__init__(out)
        self.source = Path(source) if source else self.dir.parent / "post_v2_r23a_coeff_guidance-2026-09-23" / "audit.json"

    def rho_row(self, snapshot, rho):
        pref_comb, pref_cos = {}, {}
        for pref, rs in snapshot["preferences"].items():
            comb, cos = [], []
            for r in rs:
                p = np.asarray(r["ppo_update_dir_out"], float)
                g = np.asarray(r["coeff_update_dir_out"], float)
                gn = np.linalg.norm(g)
                guide = np.zeros_like(g) if gn < EPS else rho * np.linalg.norm(p) * g / (gn + EPS)
                total = p + guide
                comb.append(total)
                cos.append(cosine(total, p))
            pref_comb[pref], pref_cos[pref] = comb, cos

        pairs, all_agg, all_suite = {}, True, True
        for a, b, name in PAIRS:
            arr = np.asarray(pref_comb[b]) - np.asarray(pref_comb[a])
            mean, target = arr.mean(0), TARGETS[name]
            suite_ok = [pair_ok(v, target) for v in arr]
            agg_ok = pair_ok(mean, target)
            all_agg &= agg_ok
            all_suite &= all(suite_ok)
            pairs[name] = {"mean_vector": mean.tolist(),
                           "sign": [sign(x) for x in mean],
                           "target_sign": [int(x) for x in target],
                           "aggregate_ok": bool(agg_ok),
                           "suite_match_fraction": float(np.mean(suite_ok)),
                           "mean_cosine_to_target": float(np.mean([cosine(v, target) for v in arr]))}
        all_cos = [x for pref in pref_cos.values() for x in pref]
        return {"pairs": pairs, "all_aggregate_ok": bool(all_agg), "all_suite_ok": bool(all_suite),
                "mean_combined_vs_ppo_cosine": float(np.mean(all_cos)),
                "min_combined_vs_ppo_cosine": float(np.min(all_cos)),
                "guide_norm_budget_ratio": rho}

    def analyze(self):
        d = json.loads(self.source.read_text())
        snaps = [{"update": s["update"],
                  "rho_candidates": {str(rho): self.rho_row(s, rho) for rho in RHO}}
                 for s in d["snapshots"]]
        first = lambda key: next(  # noqa: E731
            (rho for rho in RHO if all(s["rho_candidates"][str(rho)][key] for s in snaps)), None)
        return {
            "schema": self.schema,
            "status": "MEASUREMENT_COMPLETE", "measurement_only": True,
            # recorded repo-relative, as the original invocation did
            "source_r23a_audit": str(self.source.relative_to(REPO)
                                     if self.source.is_absolute() else self.source),
            "source_r23a_sha256": self.sha(self.source),
            "rho_candidates": list(RHO),
            "snapshots": snaps,
            "minimum_all_snapshot_aggregate_orientation_rho": first("all_aggregate_ok"),
            "minimum_all_snapshot_all_suite_orientation_rho": first("all_suite_ok"),
            "acceptance_rule": {
                "orientation": "PB/PE/BE aggregate signs correct at updates 0/1/5/10",
                "budget": "guide norm fixed to rho * ||PPO output-pressure|| per preference/suite",
                "ppo_preservation": "report combined-vs-PPO cosine; no training authorization in this audit"},
            "note": "Pure read-only algebra on R2.3-A coefficient-output pressures. No optimizer/model updates."}

    def summarize(self, rep):
        print(json.dumps({"status": rep["status"],
                          "min_aggregate_rho": rep["minimum_all_snapshot_aggregate_orientation_rho"],
                          "min_all_suite_rho": rep["minimum_all_snapshot_all_suite_orientation_rho"]},
                         indent=2))

    @classmethod
    def main(cls):
        args = cls.parse_args((("--r23a-audit",), {"help": "the R2.3-A audit to read"}))
        cls(args.out, getattr(args, "r23a_audit", None)).execute()


if __name__ == "__main__":
    NormalizedGuidanceAudit.main()
