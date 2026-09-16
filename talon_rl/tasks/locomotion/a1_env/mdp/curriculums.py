"""Custom curriculum function this A1 task needs beyond Isaac Lab's stock
isaaclab.envs.mdp.curriculums / isaaclab_tasks' velocity-task mdp module.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter


def terrain_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Adapted from isaaclab_tasks' own locomotion/velocity/mdp/curriculums.py
    terrain_levels_vel -- that stock version reads
    env.command_manager.get_command("base_velocity"), which doesn't exist
    here (this task manages v_command_buf directly, see a1_env.py's
    load_managers() -- no CommandsCfg/CommandManager wired into this env).
    Same promote/demote logic, reading env.v_command_buf instead."""
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.v_command_buf
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrain
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrain
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())
