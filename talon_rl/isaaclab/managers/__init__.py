"""Constraint-based termination manager, vendored from
jaykorea/Isaac-RL-Two-wheel-Legged-Bot's
lab/flamingo/isaaclab/isaaclab/managers/. See
docs/superpowers/specs/2026-09-15-constraint-manager-vendor-design.md for
what changed from the vendored source and why. Not wired into any env yet
— ConstraintManager has no consumer in this repo.
"""

from .constraint_term_cfg import ConstraintTermCfg

__all__ = ["ConstraintTermCfg"]
