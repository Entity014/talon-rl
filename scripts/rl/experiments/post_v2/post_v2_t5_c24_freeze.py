#!/usr/bin/env python3
"""Freeze C24: the Tracking-head residual under the reset-diverse repair."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rl.core.freeze import Freeze

SRC = "scripts/rl/experiments/post_v2"


class C24Freeze(Freeze):
    """Freeze C24: the Tracking-head residual under the reset-diverse repair."""

    run = "post_v2_t5_c24_tracking_residual-2026-09-23"
    schema = "t5_c24_tracking_residual_synthesis_v1"
    status = "C24_CLOSED_TRACKING_RESIDUAL_NONSTRUCTURAL_SHORT_HORIZON_TARGET_GEOMETRY"
    artifacts = ("audit.json", "reset_breakdown.json", "synthesis.json")

    def body(self):
        a = self.load("audit.json")
        r = self.load("reset_breakdown.json")
        t = a["aggregate"]["target_stats"]["Tracking"]
        f = a["aggregate"]["fit_summary"]["Tracking"]
        return {
            "evidence": {
                "tracking_h32_mc64_corr": t["h32_mc64_corr"],
                "tracking_h32_var": t["h32_var"],
                "tracking_mc64_var": t["mc64_var"],
                "tracking_h32_ridge1_ev": f["h32_ridge_sweep"]["1.0"]["h32_ev_mean"],
                "tracking_h32_best_sweep_ev": max(v["h32_ev_mean"] for v in f["h32_ridge_sweep"].values()),
                "tracking_mc64_ridge1_ev": f["mc64_ridge1_ev_mean"],
                "reset_breakdown": r["aggregate"],
            },
            "decision": {
                "C24": "CLOSED",
                "shared_body_tracking_capacity": "SUFFICIENT but H32 predictability weak",
                "ridge_tuning": "NOT JUSTIFIED",
                "tracking_h32_residual": "NONSTRUCTURAL / horizon-target-sensitive",
                "critic_repair_principle": "RETAIN C23 reset-diverse support",
                "actor_updates": "AUTHORIZED for next pilot only",
                "full_T4": "STILL BLOCKED",
                "V2": "OFF",
                "next": "C25 actor-updating reset-diverse critic-support pilot. Keep C23 representative reset/command/state support, shared frozen critic body, ridge lambda=1, H32 target and PPO semantics fixed; re-enable actor updates only. Primary gate: A/O semantic gradients and fresh value generalization must remain stable under policy shift.",
            },
        }

    def manifest_extra(self):
        return {"sources": {
            "main": {"sha256": self.sha(f"{SRC}/post_v2_t5_c24_tracking_residual_audit.py")},
            "reset": {"sha256": self.sha(f"{SRC}/post_v2_t5_c24_reset_breakdown.py")},
        }}

    def summary(self, syn):
        return {"status": syn["status"], "evidence": syn["evidence"], "next": syn["decision"]["next"]}


if __name__ == "__main__":
    C24Freeze.main()
