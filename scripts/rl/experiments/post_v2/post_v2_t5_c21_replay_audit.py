#!/usr/bin/env python3
"""Value replay for C21 recent3 versus recent12 support."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rl.experiments.shared.replay_audit import ReplayAudit


class C21ReplayAudit(ReplayAudit):
    """Value replay for C21 recent3 versus recent12 support."""

    arms = {"recent3": "post_v2_t5_c21_recent3-2026-09-23",
            "recent12": "post_v2_t5_c21_recent12-2026-09-23"}
    schema = "t5_c21_replay_audit_v1"
    seed_base = 1210000
    run = "post_v2_t5_c21_recent12-2026-09-23"


if __name__ == "__main__":
    C21ReplayAudit.main()
