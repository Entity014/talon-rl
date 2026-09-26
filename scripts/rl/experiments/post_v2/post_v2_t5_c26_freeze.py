#!/usr/bin/env python3
"""Freeze C26: angular-credit stability under actor learning."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rl.core.freeze import RUNS, Freeze

C25 = "post_v2_t5_c25_actor_updating25-2026-09-23"


class C26Freeze(Freeze):
    """Freeze C26: angular-credit stability under actor learning."""

    run = "post_v2_t5_c26_angular_credit-2026-09-23"
    schema = "c26_angular_credit_stability_synthesis_v1"
    status = "C26_CLOSED_ANGULAR_TRANSIENT_LOW_CONSENSUS_NOT_CRITIC_FAILURE"
    artifacts = ("audit.json", "consensus.json", "synthesis.json")

    def body(self):
        rep = self.load("replay_audit25.json", C25)
        aud = self.load("audit.json")
        return {
            "evidence": {
                "value_u25": rep["aggregate"]["25"],
                "safety_u25": self.load("u25_safety_confirm.json", C25),
                "semantic_u25": self.load("semantic_u25.json", C25),
                "angular_stability": {k: v["summary"] for k, v in aud["snapshots"].items()},
                "consensus": self.load("consensus.json")["snapshots"],
            },
            "decision": {
                "C25_durability": "critic repair PASS through u25",
                "angular_credit": "TRANSIENT instability; recovered by u25",
                "critic_failure": "REJECTED",
                "smoothness_safety": "REMAINING blocker",
                "full_T4": "BLOCKED",
                "V2": "OFF",
                "next": "C27 Smoothness safety residual audit at u10/u25: isolate failed reset suite physical trajectory and determine whether safety loss is branch-specific actor behavior or evaluation noise; no critic changes.",
            },
        }

    def manifest_extra(self):
        return {"c25_refs": {
            "replay25": {"sha256": self.sha(RUNS / C25 / "replay_audit25.json")},
            "safety25": {"sha256": self.sha(RUNS / C25 / "u25_safety_confirm.json")},
            "semantic25": {"sha256": self.sha(RUNS / C25 / "semantic_u25.json")},
        }}

    def summary(self, syn):
        return {"status": syn["status"], "next": syn["decision"]["next"]}


if __name__ == "__main__":
    C26Freeze.main()
