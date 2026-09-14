# talon_rl/tasks/locomotion/a1_env/mdp/__init__.py
"""Term functions for the A1 task — re-exports Isaac Lab's own builtins
(isaaclab.envs.mdp: joint_pos, joint_vel, last_action, time_out,
reset_scene_to_default, JointPositionActionCfg, ...) plus this task's own
observations.py and terminations.py, matching the reference repo's mdp
package convention (jaykorea/Isaac-RL-Two-wheel-Legged-Bot,
isaaclab_tasks' own cartpole/mdp/__init__.py)."""

from isaaclab.envs.mdp import *  # noqa: F401, F403

from .observations import foot_contact_binary, roll_pitch, v_command  # noqa: F401
from .terminations import obstacle_reached  # noqa: F401
