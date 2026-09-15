# talon_rl/tasks/locomotion/a1_env/a1_env_cfg.py
"""IsaacLabTalonEnvCfg(ManagerBasedRLEnvCfg) — scene + observations +
actions + terminations + events, composed from mdp/*.py term functions per
Isaac Lab's manager-based convention (mirrors isaaclab_tasks' own
cartpole_env_cfg.py and jaykorea's velocity_env_cfg.py, both verified
2026-09-14 against the real installed isaaclab 0.48.0).

RewardsCfg is deliberately empty — see a1_env.py's step() override and the
design doc's Decision 2: this repo needs an unsummed 5-term reward vector,
which RewardManager's scalar-sum contract can't produce, so reward
computation happens directly in step() via
talon_rl.reward.compute_reward_vector() instead of through this manager.

The flat ground plane is replaced with A1_ROUGH_TERRAINS_CFG
(terrain_config/rough_config.py) — climbable obstacles and a descendable
pit, both present from the start per 00_Proposal §3.3.1's Terrain
Curriculum requirement. See
docs/superpowers/specs/2026-09-15-a1-terrain-curriculum-design.md.
"""

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from talon_rl.assets.unitree_a1.a1 import TALON_A1_CFG
from talon_rl.config import ActionSpaceCfg, ObservationSpaceCfg

from . import mdp
from .terrain_config import A1_ROUGH_TERRAINS_CFG

# Set by Task 1's empirical VRAM sizing (2026-09-14) — replace this literal
# if Task 1 found a different value fits the RTX 3070 Ti's 8GB better.
_DEFAULT_NUM_ENVS = 4096  # Task 1's empirical result (2026-09-14): 4096 fits the
# RTX 3070 Ti's 8GB with ~1.64GB free at peak (6513/8192 MiB used); 8192 was
# not attempted (outside the tested decision tree, and the headroom trend
# argued against it) — see task-1-report.md in this plan's SDD workspace.


@configclass
class A1SceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=A1_ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = MISSING  # set in IsaacLabTalonEnvCfg.__post_init__

    contact_sensor = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*_foot", history_length=1)
    trunk_contact_sensor = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/trunk", history_length=1)

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=1.0)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        roll_pitch = ObsTerm(func=mdp.roll_pitch)
        foot_contact = ObsTerm(func=mdp.foot_contact_binary)
        last_action = ObsTerm(func=mdp.last_action)
        v_command = ObsTerm(func=mdp.v_command)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    obstacle_reached = DoneTerm(func=mdp.obstacle_reached)
    # Fall detection — without this, a fallen/tipped-over robot just keeps
    # accumulating steps (and reward/obs noise) until time_out instead of
    # ending the episode, diluting the training signal. Matches Isaac Lab's
    # own reference velocity locomotion template exactly (base_contact,
    # threshold=1.0) — verified against installed isaaclab 0.48.0
    # (isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py),
    # body name "trunk" swapped in for A1's own base link (vs. the
    # template's generic "base").
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("trunk_contact_sensor", body_names="trunk"), "threshold": 1.0},
    )


@configclass
class EventCfg:
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class RewardsCfg:
    """Deliberately empty — see module docstring."""


@configclass
class IsaacLabTalonEnvCfg(ManagerBasedRLEnvCfg):
    scene: A1SceneCfg = A1SceneCfg(num_envs=_DEFAULT_NUM_ENVS, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        # .replace() with no actual changes, not a direct assignment: TALON_A1_CFG
        # is a shared module-level object, and every IsaacLabTalonEnvCfg()
        # instance (e.g. Task 6's own structural test constructs more than
        # one) needs its own copy, or mutating one instance's scene.robot
        # would leak into every other instance sharing the same object.
        self.scene.robot = TALON_A1_CFG.replace()

        self.decimation = 1
        self.episode_length_s = 200 * 0.02  # matches the 2026-09-13 single-env horizon=200, dt=0.02
        self.sim.dt = 0.02
        self.sim.render_interval = self.decimation

        _obs_cfg = ObservationSpaceCfg()
        _action_cfg = ActionSpaceCfg()
        self.obs_dim = _obs_cfg.total_dim - _obs_cfg.preference_dim
        self.action_dim = _action_cfg.dim
