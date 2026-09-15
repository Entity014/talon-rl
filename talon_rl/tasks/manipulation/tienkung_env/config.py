# talon_rl/tasks/manipulation/tienkung_env/config.py
"""MDP configuration dataclasses for TienKung's bimanual box-carry task.

Same shape as talon_rl/config.py's ObservationSpaceCfg/ActionSpaceCfg/
RewardVectorCfg (this thesis's own A1 locomotion config), own numbers —
deliberately NOT the same file, since talon_rl/config.py mirrors this
thesis's own chapter3.tex tables specifically. PreferenceCfg is not
redefined here: talon_rl.config.PreferenceCfg is already dim-agnostic
(operates on reward_cfg.dim), so this task imports that one directly. See
docs/superpowers/specs/2026-09-15-tienkung-manipulation-design.md.

Prelim-scope note (mirrors talon_rl/config.py's own): no adaptation-module-
derived latent is part of this observation — the box's mass/fragility
isn't observable here. Inferring it online via the same Adaptation Module
mechanism this thesis already defines for the A1's payload mass is a later
increment.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ObservationSpaceCfg:
    """Prelim observation layout. Total dim = sum of the fields below."""

    joint_pos_dim: int = 8  # 4 joint types (shoulder_pitch/roll/yaw, elbow_pitch) x 2 arms
    joint_vel_dim: int = 8
    box_relative_pos_dim: int = 3  # box position relative to torso frame
    arm_contact_dim: int = 2  # binarized L/R arm-box contact
    prev_action_dim: int = 8
    preference_dim: int = 5  # w — one weight per reward-vector term

    @property
    def total_dim(self) -> int:
        return (
            self.joint_pos_dim
            + self.joint_vel_dim
            + self.box_relative_pos_dim
            + self.arm_contact_dim
            + self.prev_action_dim
            + self.preference_dim
        )


@dataclass(frozen=True)
class ActionSpaceCfg:
    """a_t: target joint angle per arm DOF, PD-converted to torque downstream."""

    dim: int = 8


@dataclass(frozen=True)
class RewardVectorCfg:
    """Same 5 term names as talon_rl.config.RewardVectorCfg (the A1's) —
    same multi-objective shape, retargeted semantics (see the design spec's
    Reward Vector table). `active` mirrors that file's pattern: kept here,
    not hardcoded, so a later round can add terms without touching
    preference/MOPPO code.
    """

    term_names: tuple[str, ...] = ("progress", "clearance", "energy", "impact", "smoothness")
    active: tuple[bool, ...] = field(default_factory=lambda: (True, True, True, True, True))

    # Reward-shaping constants — rough starting points, not tuned (no
    # training has run yet), same posture as talon_rl/config.py's own.
    progress_std: float = 0.3  # exp-kernel std for box-height tracking (meters)

    @property
    def dim(self) -> int:
        return len(self.term_names)
