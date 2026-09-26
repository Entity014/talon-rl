#!/usr/bin/env python3
"""Freeze C23: the reset-diverse support principle."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rl.core.freeze import Freeze

PILOT = "scripts/rl/experiments/post_v2/post_v2_t5_c23_reset_diverse_pilot.py"


class C23Freeze(Freeze):
    """Freeze C23: the reset-diverse support principle."""

    run = "post_v2_t5_c23_reset_diverse-2026-09-23"
    schema = "t5_c23_reset_diverse_synthesis_v1"
    status = "C23_CLOSED_RESET_DIVERSE_SUPPORT_PRINCIPLE_PASS_ALL_HEAD_ROBUSTNESS_PARTIAL"
    artifacts = ("audit.json", "synthesis.json")

    def body(self):
        return {
            "aggregate": self.load("audit.json")["aggregate"],
            "findings": {
                "primary": "At identical support count, ridge lambda, frozen actor/body and H32 target, independent reset/seeded support changes fresh H32 EV from -1.211 to +0.196 and Orientation from -1.142 to +0.236.",
                "mc64": "Fresh MC64 EV changes from -0.249 to +0.464; reset-diverse has 0% negative MC64 head evaluations and survival 1.0.",
                "round_consistency": "Reset-diverse H32 mean remains positive in all four rounds (+0.233,+0.159,+0.141,+0.252); Orientation is also positive in all four rounds.",
                "mechanism": "Head-solution drift is not materially lower (2.239 vs 2.184 excluding first round). The repair therefore comes primarily from representative support/generalization, not merely a more stationary optimum.",
                "residual": "Tracking remains the weak head: mean H32 EV about -0.092 with high negative fraction, while Angular, Orientation and Smoothness are positive. Thus the support principle passes but all-head robustness is not yet complete.",
            },
            "decision": {
                "C23": "CLOSED — reset-diverse support principle PASS; all-head robustness PARTIAL",
                "representative_reset_command_state_support": "SUPPORTED",
                "separate_critics": "NOT JUSTIFIED",
                "body_anchor": "NOT JUSTIFIED",
                "actor_updates": "STILL OFF for one more gate",
                "full_T4": "BLOCKED",
                "V2": "OFF",
                "next": "C24 Tracking-head residual audit under the reset-diverse repair. Keep the C23 support mechanism fixed and diagnose why Tracking H32 EV remains slightly negative while A/O/S generalize. First determine whether this is target variance/horizon mismatch or a head-specific feature-fit issue before re-enabling actor updates.",
            },
            "provenance": {
                "audit_sha256": self.sha(self.dir / "audit.json"),
                "pilot_script_sha256": self.sha(PILOT),
            },
        }

    def manifest_extra(self):
        return {"sources": {"pilot": {"sha256": self.sha(PILOT)}}}

    def summary(self, syn):
        return {"status": syn["status"], "aggregate": syn["aggregate"], "next": syn["decision"]["next"]}


if __name__ == "__main__":
    C23Freeze.main()
