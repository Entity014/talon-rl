"""MDP configuration dataclasses.

These mirror tables 3.1 (Observation Space), 3.2 (Action Space), and 3.3
(Reward Vector) in chapters/chapter3.tex of the thesis. Dimensions for the
base proprioceptive state, action, and RMA-derived quantities are taken
directly from Kumar et al. 2021 (RMA) since this thesis's Adaptation Module
extends that architecture — see \\cite{kumar2021} in chapter3.tex.

Prelim-scope note: the Adaptation Module ($\\hat z_t, \\sigma_t$) and the
Exteroception embedding are NOT part of this prelim's observation — both are
marked [TBD] in the thesis until the full pipeline is built. The prelim
observation is base proprioception + previous action + command + preference
vector only, which is enough to test the Multi-Objective Module in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ObservationSpaceCfg:
    """Prelim observation layout. Total dim = sum of the fields below."""

    joint_pos_dim: int = 12  # q  (RMA x_t component)
    joint_vel_dim: int = 12  # qdot (RMA x_t component)
    roll_pitch_dim: int = 2  # theta (RMA x_t component)
    foot_contact_dim: int = 4  # g, binarized (RMA x_t component)
    prev_action_dim: int = 12  # a_{t-1}, feedback
    command_dim: int = 3  # v_x, v_y, omega_z target
    preference_dim: int = 5  # w — one weight per reward-vector term (table 3.3)

    @property
    def total_dim(self) -> int:
        return (
            self.joint_pos_dim
            + self.joint_vel_dim
            + self.roll_pitch_dim
            + self.foot_contact_dim
            + self.prev_action_dim
            + self.command_dim
            + self.preference_dim
        )


@dataclass(frozen=True)
class ActionSpaceCfg:
    """a_t: target joint angle, PD-converted to torque downstream. R^12 per RMA."""

    dim: int = 12


@dataclass(frozen=True)
class RewardVectorCfg:
    """Table 3.3 term names, in the fixed order used everywhere in this repo.

    `active` controls which terms are actually summed in `reward.compute_reward_vector`
    — kept here (not hardcoded) so a future full-scope run can add terms without
    touching the MOPPO/preference code.
    """

    term_names: tuple[str, ...] = ("progress", "clearance", "energy", "impact", "smoothness")
    active: tuple[bool, ...] = field(default_factory=lambda: (True, True, True, True, True))

    # Reward-shaping constants (rough starting points, not tuned — expect to
    # retune once running on the real terrain curriculum, not the dummy env).
    progress_std: float = 0.5  # exp-kernel std for velocity tracking
    impact_floor_eps: float = 0.05  # w_impact >= eps, per chapter3.tex §3.2.3

    @property
    def dim(self) -> int:
        return len(self.term_names)


@dataclass(frozen=True)
class PreferenceCfg:
    """Dirichlet sampling + rate-limiting for the preference vector w (chapter3.tex fig 3.3)."""

    dirichlet_alpha: float = 1.0
    max_delta_per_step: float = 0.05  # rate-limiter cap on ||w_t - w_{t-1}||
