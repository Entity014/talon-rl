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

import dataclasses
from dataclasses import MISSING, field

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.spawners.wrappers import MultiUsdFileCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from talon_rl.assets.unitree_a1 import TALON_ASSETS_DATA_DIR
from talon_rl.assets.unitree_a1.a1 import TALON_A1_CFG
from talon_rl.config import ActionSpaceCfg, ExtrinsicsCfg, ObservationSpaceCfg

from . import mdp
from .terrain_config import A1_ROUGH_TERRAINS_CFG

# Leg-length extrinsics factor is a geometry change Isaac Lab can't
# scale-randomize on a live Articulation, so it's baked into 5 pre-scaled USD
# variants at spawn time instead (Task 5's offline generator) — random_choice
# picks one per env, independent of the runtime DR events below.
_LEG_SCALE_VARIANTS = (0.85, 0.925, 1.0, 1.075, 1.15)

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

    @configclass
    class PrivilegedCfg(ObsGroup):
        """Extrinsics e_t for the RMA Env Factor Encoder (chapter3.tex
        §3.2.1) — teacher-only, never exposed to PolicyCfg. `payload` is
        added conditionally in IsaacLabTalonEnvCfg.__post_init__, not here,
        since it depends on ExtrinsicsCfg.payload_treatment."""

        friction = ObsTerm(func=mdp.friction_extrinsic)
        motor_power = ObsTerm(func=mdp.motor_power_extrinsic)
        leg_length = ObsTerm(func=mdp.leg_length_extrinsic)
        joint_range = ObsTerm(func=mdp.joint_range_extrinsic)
        terrain_height = ObsTerm(func=mdp.local_terrain_height)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()


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

    # Domain-randomization terms for the RMA extrinsics e_t (chapter3.tex
    # §3.2.1) — everything except leg length (spawn-time USD variant, see
    # A1SceneCfg.robot below) and terrain height (already randomized by
    # which sub-terrain cell a lane spawns on).
    randomize_payload_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
            "mass_distribution_params": (0.0, 5.0),  # kg, matches Pipeline_Summary.md's Pareto-sweep table
            "operation": "add",
            "recompute_inertia": True,
        },
    )
    randomize_payload_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        # startup (not reset): randomize_rigid_body_com has no default_com buffer to
        # restore from first, so it does coms[...] += rand_samples on the CURRENT
        # value. At mode="reset" this random-walks the CoM every episode. Sampling
        # once at spawn still gives every env its own offset without accumulating.
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.02, 0.02)},  # [TBD] placeholder, not tuned
        },
    )
    randomize_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.4, 1.2),  # [TBD] placeholder, not tuned
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )
    randomize_motor_power = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "stiffness_distribution_params": (0.8, 1.2),  # [TBD] placeholder, not tuned
            "operation": "scale",
        },
    )
    randomize_joint_range = EventTerm(
        func=mdp.randomize_joint_range,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot"), "scale_range": (0.8, 1.0)},  # [TBD] placeholder, not tuned
    )

    # Dynamic perturbation (not one of the 7 RMA extrinsics -- those are static
    # properties, this is an external disturbance) so the policy learns to
    # recover from shoves instead of only ever seeing steady-state contact.
    # Matches Isaac Lab's own reference velocity locomotion template exactly
    # (isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py).
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )

    # v_command_buf (a1_env.py's load_managers()) was fixed at [0.5, 0, 0] for
    # the whole run -- progress_reward could never reward tracking a lateral
    # or turning command since none was ever commanded. [TBD] placeholder
    # ranges, not tuned.
    randomize_velocity_command = EventTerm(
        func=mdp.randomize_velocity_command,
        mode="reset",
        params={
            "lin_vel_x_range": (-0.3, 1.0),  # includes stand-still and backward, not just forward
            "lin_vel_y_range": (-0.3, 0.3),
            "ang_vel_z_range": (-0.5, 0.5),
        },
    )


@configclass
class RewardsCfg:
    """Deliberately empty — see module docstring."""


@configclass
class CurriculumCfg:
    # A1_ROUGH_TERRAINS_CFG's curriculum=False (terrain_config/rough_config.py)
    # meant every env samples terrain difficulty uniformly from the start --
    # a robot could spawn straight into a pit or tall stairs before it had
    # learned to stand on flat ground, with no way to earn its way up from
    # something easier. This term promotes/demotes each env's terrain row by
    # how far it actually walked -- see mdp.terrain_levels_vel's docstring for
    # why it isn't Isaac Lab's own stock isaaclab_tasks mdp.terrain_levels_vel
    # verbatim (that one needs a CommandManager this task doesn't have).
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


@configclass
class IsaacLabTalonEnvCfg(ManagerBasedRLEnvCfg):
    scene: A1SceneCfg = A1SceneCfg(num_envs=_DEFAULT_NUM_ENVS, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    extrinsics_cfg: ExtrinsicsCfg = field(default_factory=ExtrinsicsCfg)

    def __post_init__(self) -> None:
        # .replace() with no actual changes, not a direct assignment: TALON_A1_CFG
        # is a shared module-level object, and every IsaacLabTalonEnvCfg()
        # instance (e.g. Task 6's own structural test constructs more than
        # one) needs its own copy, or mutating one instance's scene.robot
        # would leak into every other instance sharing the same object.
        # Carry over every field from TALON_A1_CFG's existing UsdFileCfg spawn
        # (activate_contact_sensors=True in particular — a bare
        # MultiUsdFileCfg(usd_path=[...]) drops it, which silently produces a
        # robot with no foot contact reporters and a RuntimeError from
        # ContactSensor at env construction) rather than hand-picking fields,
        # since MultiUsdFileCfg is a strict superset of UsdFileCfg (just a
        # list usd_path + random_choice) and any future field added upstream
        # to UsdFileCfg should carry over too.
        _base_spawn_fields = {
            f.name: getattr(TALON_A1_CFG.spawn, f.name)
            for f in dataclasses.fields(TALON_A1_CFG.spawn)
            # usd_path: becomes the list below. func: UsdFileCfg's is the
            # single-file spawn_from_usd; MultiUsdFileCfg needs its own
            # multi-asset spawn function default, not the copied single-file one.
            if f.name not in ("usd_path", "func")
        }
        self.scene.robot = TALON_A1_CFG.replace(
            spawn=MultiUsdFileCfg(
                **_base_spawn_fields,
                usd_path=[
                    f"{TALON_ASSETS_DATA_DIR}/Robots/unitree_a1/unitree_a1_leg_scale_{s}.usd"
                    for s in _LEG_SCALE_VARIANTS
                ],
                random_choice=True,
            )
        )

        # Same sharing hazard as scene.robot above: A1SceneCfg.terrain's
        # terrain_generator field holds a reference to the module-level
        # A1_ROUGH_TERRAINS_CFG singleton, not a private copy -- mutating
        # curriculum on it in place would leak into every other
        # IsaacLabTalonEnvCfg() instance (and the module-level object
        # itself). .replace() at both levels avoids that.
        self.scene.terrain = self.scene.terrain.replace(
            terrain_generator=self.scene.terrain.terrain_generator.replace(curriculum=True)
        )

        # payload is observed+rewarded only under the "explicit" treatment —
        # under noise_only it's excluded from the privileged group entirely
        # (see ExtrinsicsCfg.dim), not just left unrewarded. Done here rather
        # than in ObservationsCfg.__post_init__ to follow this file's existing
        # post-construction mutation pattern (self.scene.robot above).
        if self.extrinsics_cfg.payload_treatment != "noise_only":
            self.observations.privileged.payload = ObsTerm(func=mdp.payload_extrinsics)

        self.decimation = 1
        self.episode_length_s = 200 * 0.02  # matches the 2026-09-13 single-env horizon=200, dt=0.02
        self.sim.dt = 0.02
        self.sim.render_interval = self.decimation

        # Default ViewerCfg is a fixed world-space camera at (7.5, 7.5, 7.5)
        # looking at world origin -- with a terrain generator (env_spacing=2.5,
        # 4096 envs) the robot is almost never anywhere near there, so
        # play.py's --video came out pointed at empty/wrong terrain with no
        # robot in frame. asset_root tracking follows env 0's robot every
        # frame instead; eye/lookat become offsets from that root, not world
        # coordinates.
        self.viewer.origin_type = "asset_root"
        self.viewer.env_index = 0
        self.viewer.asset_name = "robot"
        self.viewer.eye = (2.0, 2.0, 1.2)
        self.viewer.lookat = (0.0, 0.0, 0.3)

        _obs_cfg = ObservationSpaceCfg()
        _action_cfg = ActionSpaceCfg()
        self.obs_dim = _obs_cfg.total_dim - _obs_cfg.preference_dim
        self.action_dim = _action_cfg.dim
