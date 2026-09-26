"""Pure tensor math for mdp.curriculums.terrain_levels_vel's promotion-dwell
requirement, split out so it's testable without importing the a1_env
package (which requires a live Isaac Sim `carb` context just to import --
see talon_rl/tasks/locomotion/a1_env/__init__.py)."""

import torch


def apply_promote_streak(streak: torch.Tensor, qualifies: torch.Tensor, required: int) -> tuple[torch.Tensor, torch.Tensor]:
    """streak: per-env consecutive-qualifying-episode counter (mutated
    in place via the returned tensor -- caller writes it back to its own
    buffer at the right env_ids). qualifies: this episode's raw
    promotion criterion (distance reached, didn't fall). Returns
    (updated_streak, move_up) -- move_up only true once streak reaches
    `required`, at which point the streak resets to 0 for the next level."""
    streak = torch.where(qualifies, streak + 1, torch.zeros_like(streak))
    move_up = qualifies & (streak >= required)
    streak = torch.where(move_up, torch.zeros_like(streak), streak)
    return streak, move_up
