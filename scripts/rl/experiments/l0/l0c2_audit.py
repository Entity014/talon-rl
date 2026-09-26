#!/usr/bin/env python3
"""What the C1 snapshots can and cannot say about lane-level geometry.

Read-only, and deliberately inconclusive: the C1 snapshots reduced action and
state arrays to batch means before serialising, so the requested lane-level
correlation is not computable from them. This reports the quantiles that do
survive and names exactly what is missing, rather than inventing lane geometry.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rl.core.offline_audit import ARTIFACTS, RUNS, OfflineAudit

RUNS_BY_SEED = {0: "l0c1_seed0_2026-09-22", 1: "l0c1_seed1_2026-09-22", 2: "l0c1_seed2_2026-09-22"}
UPDATES = (200, 300, 400, 500)
CONCLUSION = (
    "Existing C1 snapshots cannot answer the requested lane-level geometry "
    "correlation because action/state arrays were reduced to batch means before "
    "serialization. A truthful C2 requires read-only checkpoint replay that emits "
    "lane-indexed arrays (and a stable lane-to-evaluator-state mapping); no "
    "training or formulation change is authorized.")


def quantiles(x):
    x = np.asarray(x, float)
    return {"n": int(x.size),
            "p50": float(np.percentile(x, 50)) if x.size else None,
            "p95": float(np.percentile(x, 95)) if x.size else None,
            "p99": float(np.percentile(x, 99)) if x.size else None,
            "max": float(x.max()) if x.size else None}


class L0C2ArtifactGranularityAudit(OfflineAudit):
    """C2 lane-geometry audit, limited by what C1 actually serialised."""

    root = ARTIFACTS
    run = "l0c2_audit"
    report = "L0C2_AUDIT.json"
    sort_keys = True
    schema = "l0c2_lane_geometry_audit_v1"

    def analyze(self):
        out = {
            "schema": self.schema,
            "status": "INCONCLUSIVE_ARTIFACT_GRANULARITY",
            "updates": UPDATES,
            "available_lane_level": ["episode_lengths for completed episodes only"],
            "aggregate_only": ["action_norm", "action_saturation", "base_contact", "height",
                               "roll", "pitch", "vx_error", "reward terms", "lane_age"],
            "missing_for_requested_c2": ["per-lane action norm/saturation",
                                         "per-joint action values",
                                         "per-lane actor mean",
                                         "per-lane tilt/height/contact/vx trajectories",
                                         "lane identity linking training snapshots to deterministic evaluator"],
            "seeds": {}}
        for s, run in RUNS_BY_SEED.items():
            out["seeds"][str(s)] = {}
            for u in UPDATES:
                snap = json.loads((RUNS / run / "snapshots" / f"update_{u:03d}.json").read_text())
                x = snap["telemetry"]
                out["seeds"][str(s)][str(u)] = {
                    "completed_episode_length_quantiles": quantiles(x["episode_lengths"]),
                    "aggregate_action_norm": x["action_norm"],
                    "aggregate_action_saturation": x["action_saturation"],
                    "aggregate_vx_error": x["vx_error"]}
        out["conclusion"] = CONCLUSION
        return out

    def summarize(self, report):
        lines = ["# L0-C2 lane-level geometry audit", "",
                 "Status: **INCONCLUSIVE — existing artifact granularity**", "",
                 report["conclusion"], "",
                 "Requested lane-level quantiles/covariance cannot be computed from C1 "
                 "snapshots without inventing data. Only completed-episode length quantiles "
                 "and batch aggregates are reported in the JSON artifact."]
        (self.out / "report.md").write_text("\n".join(lines) + "\n")
        print(self.out / self.report)


if __name__ == "__main__":
    L0C2ArtifactGranularityAudit.main()
