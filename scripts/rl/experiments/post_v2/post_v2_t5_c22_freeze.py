#!/usr/bin/env python3
"""Freeze C22: support-topology diversity."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rl.core.freeze import RUNS, Freeze

CONTROL = "post_v2_t5_c21_recent12warm-2026-09-23"


class C22Freeze(Freeze):
    """Freeze C22: support-topology diversity."""

    run = "post_v2_t5_c22_diverse12-2026-09-23"
    schema = "t5_c22_support_topology_diversity_synthesis_v1"
    status = "C22_CLOSED_DIVERSITY_STRONGLY_HELPFUL_BUT_CURRENT_STREAM_DIVERSE12_NOT_SUFFICIENT"
    artifacts = ("audit.json", "replay_audit.json", "terminal_multisuite.json", "synthesis.json")

    def body(self):
        rep = self.load("replay_audit.json")
        mul = self.load("terminal_multisuite.json")
        return {
            "aggregate": {
                "coverage": {
                    "consecutive_support_to_seen_distance_mean": 2.1288670668,
                    "diverse_support_to_seen_distance_mean": 0.0608117817,
                    "diverse_over_consecutive_ratio": 0.028565326,
                    "diverse_support_age_span_mean": 18.42857143,
                },
                "head_solution_drift": {"consecutive12": 1.027003399, "diverse12": 0.625231252},
                "replay_u25": {
                    "consecutive12": rep["aggregate"]["consecutive12"][25],
                    "diverse12": rep["aggregate"]["diverse12"][25],
                },
                "terminal_multisuite": mul["aggregate"],
            },
            "decision": {
                "C22": "CLOSED — diversity hypothesis SUPPORTED; sufficiency gate FAIL",
                "support_diversity": "STRONGLY SUPPORTED",
                "diverse12_within_single_stream": "INSUFFICIENT as final repair",
                "actor_updates": "REMAIN OFF",
                "full_T4": "BLOCKED",
                "V2": "OFF",
                "next": "C23 true reset-diverse reference-support pilot with exactly 12 independently reset/command-seeded supports; actor/body frozen, ridge lambda=1, H32 target. Primary gate: multi-suite fresh H32 EV non-negative, especially Orientation.",
            },
            "provenance": {
                "control_audit_sha256": self.sha(RUNS / CONTROL / "audit.json"),
                "diverse_audit_sha256": self.sha(self.dir / "audit.json"),
                "replay_sha256": self.sha(self.dir / "replay_audit.json"),
                "multisuite_sha256": self.sha(self.dir / "terminal_multisuite.json"),
            },
        }

    def manifest_extra(self):
        return {"paired_control": {"audit.json": {"sha256": self.sha(RUNS / CONTROL / "audit.json")}}}

    def summary(self, syn):
        return {"status": syn["status"],
                "multisuite": syn["aggregate"]["terminal_multisuite"],
                "next": syn["decision"]["next"]}


if __name__ == "__main__":
    C22Freeze.main()
