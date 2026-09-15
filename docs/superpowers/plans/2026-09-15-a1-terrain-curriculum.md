# A1 Rough Terrain (Climbable + Pit) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the A1 env's flat ground plane with a `TerrainGeneratorCfg` that includes climbable obstacles (stairs, boxes) and a descendable pit from the start, per the thesis proposal's CORE Terrain Curriculum requirement (§3.3.1).

**Architecture:** One new `talon_rl/tasks/locomotion/a1_env/terrain_config/` package holding `A1_ROUGH_TERRAINS_CFG` (an `isaaclab.terrains.TerrainGeneratorCfg`), wired into `A1SceneCfg` in `a1_env_cfg.py` via a `TerrainImporterCfg` replacing the current flat `GroundPlaneCfg`. No curriculum progression, no held-out terrain set — those are later increments (see spec's Out of Scope).

**Tech Stack:** Python 3.12, Isaac Lab (`isaaclab.terrains`), pytest (GPU-gated).

**Spec:** [docs/superpowers/specs/2026-09-15-a1-terrain-curriculum-design.md](../specs/2026-09-15-a1-terrain-curriculum-design.md)

## Global Constraints

- Do NOT change any reward, observation, or termination term — the proposal's emergent-behavior claim (§3.3.1) depends on the existing 5-term reward vector (`talon_rl/reward.py`) staying exactly as-is; this is terrain-only.
- Both required sub-terrain categories (climbable obstacles, descendable pit) must have proportion > 0 in `A1_ROUGH_TERRAINS_CFG.sub_terrains` — all proportions across `sub_terrains` must sum to 1.0 (`TerrainGeneratorCfg` requirement).
- No curriculum progression this round: `TerrainGeneratorCfg(curriculum=False, ...)`.
- This machine has no Isaac Lab installed anywhere (confirmed via `pip show isaaclab` and a filesystem search) — nothing in this plan can be executed or its tests run in this environment. Every field name taken from Isaac Lab's API (`MeshPitTerrainCfg`'s exact fields, `TerrainImporterCfg`'s exact fields, `InteractiveScene`'s asset-lookup syntax) must be verified against the actually-installed `isaaclab` package on a real GPU machine before/while writing the code — treat every Isaac Lab API surface below as "best available knowledge, confirm against the installed package," the same posture `a1_env_cfg.py`'s own module docstring already documents for its existing imports (verified 2026-09-14 against isaaclab 0.48.0).
- Testing follows the existing GPU-gated pattern (`pytest.importorskip("isaacsim")`, like the rest of `tests/test_a1_env.py`) — do not attempt to mock `isaaclab.terrains` the way `talon_rl.isaaclab.managers`'s tests mock `isaaclab.managers` (procedural terrain mesh generation isn't reasonably mockable; see the terrain design spec's Testing section for why).

---

### Task 1: Terrain generator + scene wiring

**Files:**
- Create: `talon_rl/tasks/locomotion/a1_env/terrain_config/__init__.py`
- Create: `talon_rl/tasks/locomotion/a1_env/terrain_config/rough_config.py`
- Modify: `talon_rl/tasks/locomotion/a1_env/a1_env_cfg.py`
- Modify: `tests/test_a1_env.py`

**Interfaces:**
- Produces: `talon_rl.tasks.locomotion.a1_env.terrain_config.A1_ROUGH_TERRAINS_CFG` — an `isaaclab.terrains.TerrainGeneratorCfg` instance with `sub_terrains` containing at least `"pyramid_stairs"`, `"pyramid_stairs_inv"`, `"boxes"` (climbable, proportions summing to 0.7) and `"pit"` (descendable, proportion 0.3).
- Consumes: nothing from earlier tasks (this is a single-task plan).

This task is not independently testable in this environment (no Isaac Lab installed anywhere on this machine) — there is no RED/GREEN cycle to run locally. Follow the steps below as written; the test added in Step 4 is real and correct, but only executable on a GPU machine with Isaac Sim installed (same situation as every existing test in `tests/test_a1_env.py`).

- [ ] **Step 1: Create the terrain generator config**

Create `talon_rl/tasks/locomotion/a1_env/terrain_config/rough_config.py`:

```python
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
```

Verify against the installed `isaaclab` package: `MeshPitTerrainCfg`'s
field names (`pit_depth_range`, `platform_width`, `double_pit` are the
names found in Isaac Lab's published API docs at design time — confirm
they match the actually-installed version before running) and that
`terrain_gen.MeshPitTerrainCfg` is reachable via the top-level
`isaaclab.terrains` namespace the same way `MeshPyramidStairsTerrainCfg`
is (it should be, per Isaac Lab's `isaaclab/terrains/__init__.py`
re-exporting everything under `isaaclab.terrains.trimesh.mesh_terrains_cfg`,
but confirm).

- [ ] **Step 2: Create the package `__init__.py`**

Create `talon_rl/tasks/locomotion/a1_env/terrain_config/__init__.py`:

```python
"""A1 terrain configurations."""

from .rough_config import A1_ROUGH_TERRAINS_CFG

__all__ = ["A1_ROUGH_TERRAINS_CFG"]
```

- [ ] **Step 3: Wire the terrain generator into `A1SceneCfg`**

In `talon_rl/tasks/locomotion/a1_env/a1_env_cfg.py`:

Add these imports (alongside the existing `isaaclab.*` imports near the
top of the file):

```python
from isaaclab.terrains import TerrainImporterCfg

from .terrain_config import A1_ROUGH_TERRAINS_CFG
```

(`from .terrain_config import A1_ROUGH_TERRAINS_CFG` goes with the
existing `from . import mdp` local-package import, not the `isaaclab.*`
block.)

Replace this line (currently the first field of `A1SceneCfg`):

```python
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
```

with:

```python
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
```

No other field in `A1SceneCfg` changes — `robot`, `contact_sensor`,
`dome_light` stay exactly as they are. Verify `TerrainImporterCfg`'s field
names (`prim_path`, `terrain_type`, `terrain_generator`,
`max_init_terrain_level`, `collision_group`, `physics_material`,
`debug_vis`) against the installed `isaaclab.terrains.TerrainImporterCfg`
— these are taken from jaykorea/Isaac-RL-Two-wheel-Legged-Bot's
`velocity_env_cfg.py` (a real, working Isaac Lab velocity task), so they
should match, but this repo's own convention (see this file's own module
docstring) is to verify Isaac Lab imports against the actually-installed
version rather than assume.

Also update this file's module docstring (top of file) to mention the
terrain change — add a short paragraph after the existing
`RewardsCfg is deliberately empty...` paragraph:

```python
The flat ground plane is replaced with A1_ROUGH_TERRAINS_CFG
(terrain_config/rough_config.py) — climbable obstacles and a descendable
pit, both present from the start per 00_Proposal §3.3.1's Terrain
Curriculum requirement. See
docs/superpowers/specs/2026-09-15-a1-terrain-curriculum-design.md.
```

- [ ] **Step 4: Add a structural terrain assertion to the existing GPU-gated test**

In `tests/test_a1_env.py`, add this import inside
`test_isaac_lab_env_implements_base_contract()`'s `try:` block, alongside
the existing `from talon_rl.tasks.locomotion.a1_env.a1_env_cfg import
IsaacLabTalonEnvCfg` line:

```python
        from talon_rl.tasks.locomotion.a1_env.terrain_config import A1_ROUGH_TERRAINS_CFG
```

Add this assertion right after the existing
`env = gym.make("Isaac-Talon-A1-v0", cfg=cfg).unwrapped` line:

```python
        # terrain: confirm the scene uses the generator config, not a flat
        # ground plane — exact InteractiveScene asset-lookup syntax
        # (env.scene["terrain"]) taken from Isaac Lab's own convention,
        # verify against the installed isaaclab.scene API
        assert env.scene["terrain"].cfg.terrain_generator is A1_ROUGH_TERRAINS_CFG
```

- [ ] **Step 5: Verify on a GPU machine with Isaac Sim installed (not this environment)**

This step cannot run here — no Isaac Lab is installed on this machine.
When run on a GPU machine with Isaac Sim installed, the command is:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. pytest tests/test_a1_env.py -v
```

Expected: `test_isaac_lab_env_implements_base_contract` PASSES, including
the new terrain assertion. If any Isaac Lab API name in Steps 1/3/4 doesn't
match the installed package (per the Global Constraints' verification
note), fix the name to match — the sub-terrain composition and scene
wiring described above is the actual requirement, not the exact spelling
of any one field.

- [ ] **Step 6: Commit**

```bash
git add talon_rl/tasks/locomotion/a1_env/terrain_config talon_rl/tasks/locomotion/a1_env/a1_env_cfg.py tests/test_a1_env.py
git commit -m "$(cat <<'EOF'
feat: add A1 rough terrain (climbable obstacles + descendable pit)

First increment of 00_Proposal §3.3.1's CORE Terrain Curriculum
requirement — replaces the flat ground plane with a TerrainGeneratorCfg
whose sub-terrains guarantee both a climbable-obstacle category and a
descendable pit from the start, so the existing 5-term reward vector
has terrain to actually express the Progress/Impact/Clearance
trade-off on (no new reward terms). No curriculum progression or
held-out terrain set yet — later increments.

Untested in this environment (no Isaac Lab installed here) — verify on
the GPU machine per the plan's Step 5 before relying on this.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```
