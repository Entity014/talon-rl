# talon_rl/tasks/locomotion/a1_env/mdp/__init__.py
"""Term functions for the A1 task — re-exports Isaac Lab's own builtins
(isaaclab.envs.mdp: joint_pos, joint_vel, last_action, time_out,
reset_scene_to_default, JointPositionActionCfg, ...) plus this task's own
observations.py and terminations.py, matching the reference repo's mdp
package convention (jaykorea/Isaac-RL-Two-wheel-Legged-Bot,
isaaclab_tasks' own cartpole/mdp/__init__.py)."""

from isaaclab.envs.mdp import *  # noqa: F401, F403

from .curriculums import terrain_levels_vel  # noqa: F401
from .events import randomize_joint_range, randomize_velocity_command  # noqa: F401
from .observations import (  # noqa: F401
    foot_contact_binary,
    friction_extrinsic,
    joint_range_extrinsic,
    leg_length_extrinsic,
    local_terrain_height,
    motor_power_extrinsic,
    payload_extrinsics,
    roll_pitch,
    v_command,
)
from .terminations import obstacle_reached  # noqa: F401
