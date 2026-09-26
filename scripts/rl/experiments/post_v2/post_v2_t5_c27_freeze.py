#!/usr/bin/env python3
"""Freeze C27: the smoothness-branch trajectory safety risk."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rl.core.freeze import Freeze

C25 = "post_v2_t5_c25_actor_updating25-2026-09-23"


class C27Freeze(Freeze):
    """Freeze C27: the smoothness-branch trajectory safety risk."""

    run = "post_v2_t5_c27_smoothness_safety-2026-09-23"
    schema = "c27_smoothness_safety_synthesis_v1"
    status = "C27_CLOSED_SMOOTHNESS_BRANCH_TRAJECTORY_SAFETY_RISK_LOCAL_GRADIENT_NOT_SUFFICIENT"
    artifacts = ("audit.json", "counterfactual.json", "synthesis.json")

    def body(self):
        a = self.load("audit.json")
        return {
            "evidence": {
                "u25_safety": self.load("u25_safety_confirm.json", C25),
                "failed_seed_trace": a["snapshots"]["25"],
                "counterfactual": self.load("counterfactual.json"),
            },
            "decision": {
                "C27": "CLOSED",
                "smoothness_safety_failure": "REAL / branch-specific / trajectory-level",
                "critic_failure": "REJECTED",
                "local_smoothness_gradient_as_root_cause": "NOT SUPPORTED",
                "full_T4": "BLOCKED",
                "V2": "OFF",
                "next": "C28 Smoothness basin/safety audit: characterize S-heavy policy state visitation and recovery margin on the failing reset versus survivors, then decide whether smoothness should remain a free MORL axis or require a safety constraint/regularizer. No critic changes.",
            },
        }

    def manifest_extra(self):
        from rl.core.freeze import RUNS
        return {"c25_refs": {"u25_safety": {"sha256": self.sha(RUNS / C25 / "u25_safety_confirm.json")}}}

    def summary(self, syn):
        return {"status": syn["status"], "next": syn["decision"]["next"]}


if __name__ == "__main__":
    C27Freeze.main()
