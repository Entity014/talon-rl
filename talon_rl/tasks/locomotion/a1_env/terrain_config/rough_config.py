# talon_rl/tasks/locomotion/a1_env/terrain_config/rough_config.py
"""Rough terrain for the A1 locomotion task — climbable obstacles (stairs,
boxes) and a descendable pit, both present from the start per the thesis
proposal's Terrain Curriculum requirement (00_Proposal §3.3.1): strategic
behaviors like climbing over an obstacle instead of detouring, or
descending into a pit slowly instead of detouring, must emerge from the
existing reward vector (talon_rl/reward.py) combined with terrain that
actually offers those choices — not from new reward terms. See
docs/superpowers/specs/2026-09-15-a1-terrain-curriculum-design.md.

No curriculum progression yet (curriculum=False below) — that, and the
separate "extreme-terrain held-out" set used in Gate 1 (proposal §3.4),
are later increments.
"""

from __future__ import annotations

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg

A1_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.5,
    use_cache=False,
    curriculum=False,
    sub_terrains={
        # climbable obstacles
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.05, 0.15),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.05, 0.15),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2,
            grid_width=0.45,
            grid_height_range=(0.02, 0.06),
            platform_width=2.0,
        ),
        # descendable pit — proposal §3.3.1's primary qualitative test
        # scenario (§3.7.2): descend slowly vs. detour, gated by
        # w_progress vs. w_impact
        "pit": terrain_gen.MeshPitTerrainCfg(
            proportion=0.3,
            pit_depth_range=(0.3, 1.0),
            platform_width=2.0,
            double_pit=False,
        ),
    },
)
"""Rough terrain generator for the A1 locomotion task: climbable
obstacles (pyramid_stairs, pyramid_stairs_inv, boxes) and a descendable
pit (pit), proportions summing to 1.0.
"""
