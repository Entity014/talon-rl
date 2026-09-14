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
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 0] >= env.obstacle_ahead_buf
