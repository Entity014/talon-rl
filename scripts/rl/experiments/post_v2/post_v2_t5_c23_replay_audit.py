#!/usr/bin/env python3
"""Value replay for C23 reset-recent versus reset-diverse support."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rl.experiments.shared.replay_audit import ReplayAudit


class C23ReplayAudit(ReplayAudit):
    """Value replay for C23 reset-recent versus reset-diverse support."""

    arms = {"reset12_recent": "post_v2_t5_c22_reset12-2026-09-23",
            "reset12_diverse": "post_v2_t5_c23_reset_diverse12-2026-09-23"}
    schema = "t5_c21_replay_audit_v1"
    seed_base = 1210000
    run = "post_v2_t5_c23_reset_diverse12-2026-09-23"


if __name__ == "__main__":
    C23ReplayAudit.main()
