#!/usr/bin/env python3
"""Target gap for C5 at one critic update against C7 at ten."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rl.experiments.shared.target_gap_audit import TargetGapAudit


class C7TargetGapAudit(TargetGapAudit):
    """Target gap for C5 at one critic update against C7 at ten."""

    run = "post_v2_t5_c7_critic_updates10-2026-09-23"
    arms = {"C5_u1": ("post_v2_t5_c5_h16-2026-09-23", None),
            "C7_u10": ("post_v2_t5_c7_critic_updates10-2026-09-23", None)}
    seed_base = 800000


if __name__ == "__main__":
    C7TargetGapAudit.main()
