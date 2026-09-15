# talon_rl/isaaclab/managers/constraint_term_cfg.py
"""Configuration for a constraint term, vendored from
jaykorea/Isaac-RL-Two-wheel-Legged-Bot's
lab/flamingo/isaaclab/isaaclab/managers/constraint_term_cfg.py. See
docs/superpowers/specs/2026-09-15-constraint-manager-vendor-design.md.
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import Callable

import torch
from isaaclab.managers.manager_term_cfg import ManagerTermBaseCfg
from isaaclab.utils import configclass


@configclass
class ConstraintTermCfg(ManagerTermBaseCfg):
    """Configuration for a constraint term."""

    func: Callable[..., torch.Tensor] = MISSING
    """The function to call for this term.

    Must take the environment object and any params in `self.params` as
    input and return a float tensor of shape (num_envs,) giving each env's
    degree of violation in [0, 1].
    """

    p_max: float = 1.0
    """Maximum scaling factor for this constraint's termination probability.

    Use 1.0 for a hard constraint (strictly enforced once the curriculum
    reaches full strength, if any). Use a value below 1.0 for a soft
    constraint that still allows some exploration even at full curriculum
    strength.
    """

    use_curriculum: bool = False
    """Whether to ramp this constraint's enforcement in over training via
    ConstraintManager's curriculum, rather than enforcing at full strength
    from the first step."""

    time_out: str = "terminate"
    """How this term's violation value is consumed: "truncate" or
    "terminate" for a hard binary 0/1 termination signal, or "constraint"
    for ConstraintManager's soft, curriculum-scaled stochastic termination
    probability. Reuses ManagerTermBaseCfg's `time_out` field name (there a
    plain bool) with three-way string semantics instead — kept for
    Isaac Lab API-shape compatibility rather than renamed to a
    constraint-specific field.
    """
