#!/usr/bin/env python3
"""Target gap at GAE lambda 0.95 against lambda 1."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rl.experiments.shared.target_gap_audit import TargetGapAudit


class C8TargetGapAudit(TargetGapAudit):
    """Target gap at GAE lambda 0.95 against lambda 1."""

    run = "post_v2_t5_c8_lambda1-2026-09-23"
    arms = {"C5_lam095": ("post_v2_t5_c5_h16-2026-09-23", .95),
            "C8_lam1": ("post_v2_t5_c8_lambda1-2026-09-23", 1.0)}
    seed_base = 800000


if __name__ == "__main__":
    C8TargetGapAudit.main()
