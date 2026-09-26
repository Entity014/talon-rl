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

from talon_rl.curricula.streak import apply_promote_streak


_PROMOTE_STREAK_REQUIRED = 3  # consecutive qualifying episodes needed before promotion, see docstring


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
    Same promote/demote logic, reading env.v_command_buf instead.

    One deviation from the stock version: `move_up` also requires the
    episode NOT to have ended via `base_contact` (a fall). The stock
    criterion is pure distance-from-origin, which this task's `pit`
    sub-terrain (pit_depth_range up to 1.0m, terrain_config/rough_config.py)
    can satisfy by tumbling/sliding several meters while falling in, rather
    than by walking there -- diagnosed 2026-09-17 (phase1_easystart run):
    mean_episode_length spiked to ~59/200 by iteration 101 then collapsed
    back to the ~13-step floor by iteration ~400 and never recovered. Still a
    correctness fix worth keeping, but NOT the cause of that floor: a
    same-day isolation run with move_up forced permanently False (terrain
    frozen at level 0 the whole run) showed the identical peak-then-collapse
    shape, ruling out curriculum-driven difficulty escalation entirely. The
    actual cause traced to ActorCritic having no action bound at all (see
    scripts/rl/core/modules/actor_critic.py's ACTION_CLIP).

    Second deviation, added 2026-09-18: promotion requires
    _PROMOTE_STREAK_REQUIRED consecutive qualifying episodes at the current
    level, not just one. The stock/single-episode version promotes on the
    very first lucky success, which can be a single noisy rollout rather
    than a consolidated skill -- a plausible contributor to the
    spike-then-collapse shape seen around iteration ~13,500 in
    phase1_longrun (mean_episode_len briefly hit 70.45 then fell straight
    back to the floor): a policy promoted to harder terrain off one success
    faces a harder distribution before the behavior that earned the
    promotion has stabilized. A single failed or short episode resets the
    streak to 0, same as before.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.v_command_buf
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    fell = env.termination_manager.get_term("base_contact")[env_ids]
    # robots that walked far enough without falling qualify for promotion
    qualifies = (distance > terrain.cfg.terrain_generator.size[0] / 2) & ~fell
    updated_streak, move_up = apply_promote_streak(
        env.terrain_promote_streak[env_ids], qualifies, _PROMOTE_STREAK_REQUIRED
    )
    env.terrain_promote_streak[env_ids] = updated_streak
    # robots that walked less than half of their required distance go to simpler terrain
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())
