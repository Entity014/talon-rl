# talon_rl/tasks/locomotion/a1_env/mdp/terminations.py
"""Termination terms. Horizon timeout uses Isaac Lab's own mdp.time_out
builtin (referenced directly in a1_env_cfg.py's TerminationsCfg) — this file
only adds the scripted obstacle check, ported from the single-env
IsaacLabTalonEnv's `pos_x >= obstacle_ahead` condition (2026-09-13), still a
placeholder pending the Exteroception Module.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def obstacle_reached(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """root_pos_w is world-frame, but the terrain grid spreads env_origins.x
    across [-border, +border] (see terrain_config/rough_config.py's
    border_width=20.0) — the single-env port compared world x straight
    against obstacle_ahead_buf, so any env whose origin.x alone exceeded 5.0
    terminated on the very first step regardless of the robot's actual
    displacement. Must subtract env_origins.x first, same as
    mdp/curriculums.py's distance-from-spawn check.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    pos_x_local = asset.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    return pos_x_local >= env.obstacle_ahead_buf
