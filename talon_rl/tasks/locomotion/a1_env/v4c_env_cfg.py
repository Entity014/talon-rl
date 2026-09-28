# talon_rl/tasks/locomotion/a1_env/v4c_env_cfg.py
"""V4-C environment: stock Isaac-Velocity-Flat-Unitree-A1-v0 plus the 12-D
privileged e_t group, nothing else (decision 2026-09-26, "R2: stock + e_t").

V4-C tests the TeacherV4 architecture on the substrate M0/V3 were trained on,
so physics (dt 0.005, decimation 4), episode length, flat terrain, command
manager, terminations, events (push off, stock base-mass range), rewards and
the action contract (scale 0.25, Kp 25 / Kd 0.5) are the stock ones, and the
robot is the repo's a1.usd, as in every M0/V3 script. e_t keeps the frozen
12-D layout; channels stock does not randomize read constant. Wider plant and
morphology variation is V4-D's treatment (Isaac-Talon-A1-v0).
"""

from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import ObservationsCfg as StockObservationsCfg

from talon_rl.assets import TALON_ASSETS_DATA_DIR

from . import mdp


@configclass
class V4CObservationsCfg(StockObservationsCfg):
    @configclass
    class PrivilegedCfg(ObsGroup):
        """Same 12-D order as IsaacLabTalonEnvCfg's privileged group
        (docs/verdicts/teacher_v4/teacher-v4-a-b-input-contract-verdict.md)."""

        friction = ObsTerm(func=mdp.friction_extrinsic)
        motor_power = ObsTerm(func=mdp.motor_power_extrinsic)
        leg_length = ObsTerm(func=mdp.nominal_leg_length)
        joint_range = ObsTerm(func=mdp.joint_range_extrinsic)
        terrain_height = ObsTerm(func=mdp.local_terrain_height)
        dynamic_friction = ObsTerm(func=mdp.dynamic_friction_extrinsic)
        joint_damping = ObsTerm(func=mdp.joint_damping_extrinsic)
        payload = ObsTerm(func=mdp.payload_extrinsics)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    privileged: PrivilegedCfg = PrivilegedCfg()


@configclass
class TalonV4CEnvCfg(UnitreeA1FlatEnvCfg):
    observations: V4CObservationsCfg = V4CObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.robot.spawn.usd_path = f"{TALON_ASSETS_DATA_DIR}/Robots/unitree_a1/a1.usd"


@configclass
class TalonV4CS1EnvCfg(TalonV4CEnvCfg):
    """V4-C2S-R1: identical to V4-C plus the action-jerk term, which replaces
    action_rate_l2 as the S objective (the stock action_rate_l2 term stays in
    the manager; the objective vector selects which one S uses). Weight
    −0.01 as for action_rate_l2; the S1 normalization divisor is measured
    separately with the T3-B protocol."""

    def __post_init__(self) -> None:
        super().__post_init__()
        from isaaclab.managers import RewardTermCfg as RewTerm
        self.rewards.action_jerk_l2 = RewTerm(func=mdp.action_jerk_l2, weight=-0.01)


@configclass
class TalonV4CV3EnvCfg(TalonV4CEnvCfg):
    """F8 V3 arm: identical to V4-C plus body_height_osc_l2, the V3 candidate
    for Vertical Stability. Weight −1.0; the objective divisor is measured
    with the T3-B protocol, so the weight's scale cancels."""

    def __post_init__(self) -> None:
        super().__post_init__()
        from isaaclab.managers import RewardTermCfg as RewTerm
        self.rewards.body_height_osc_l2 = RewTerm(func=mdp.body_height_osc_l2, weight=-1.0)
