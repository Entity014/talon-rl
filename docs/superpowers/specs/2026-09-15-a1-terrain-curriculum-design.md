# A1 rough terrain (climbable obstacles + gap/pit) — first terrain generator (design)

## Context

The thesis proposal (`talon-thesis/00_Proposal/Latex/proposal.pdf`, §3.3.1)
states a CORE requirement: strategic behaviors like climbing over an
obstacle instead of detouring, or descending into a gap/pit slowly instead
of detouring, must **emerge** from the existing 5-term reward vector
(Progress, Obstacle clearance, Energy, Impact, Smoothness — see
`talon_rl/reward.py`) when combined with the right terrain — not from new
reward terms. The proposal is explicit that the terrain curriculum must
include both climbable obstacles **and** gaps/pits **from the start of
training**, or the policy never gets the chance to learn the
Progress/Impact/Clearance trade-off no matter how the reward is designed.
§3.7.2 additionally names the gap/pit terrain as the primary **qualitative
test scenario for Chapter 4**: at high `w_progress`/low `w_impact` the
policy is expected to descend fast and accept risk; at high
`w_impact`/low `w_progress`, to detour or descend slowly. This is the main
piece of evidence used in the defense to show preference-conditioning
works on intuition, not just numbers in a table.

`talon_rl`'s A1 env currently spawns a flat `sim_utils.GroundPlaneCfg()`
(`talon_rl/tasks/locomotion/a1_env/a1_env_cfg.py:45`) — no terrain
generation exists yet. This is the first increment of that work.

Mirrors jaykorea/Isaac-RL-Two-wheel-Legged-Bot's
`tasks/.../velocity/terrain_config/rough_config.py` pattern (a
`TerrainGeneratorCfg` with named sub-terrains at chosen proportions), not
copied verbatim — their config has no gap/pit sub-terrain, and this one
must.

## Scope

**In scope**: one `TerrainGeneratorCfg` with sub-terrains covering both
required categories (climbable obstacles, descendable gap/pit) from
proportion > 0, wired into `A1SceneCfg` in place of the flat ground plane.

**Out of scope** (later increments, not designed here):
- Curriculum progression / difficulty ramping over training
  (`TerrainImporterCfg(curriculum=...)` stays `False` this round)
- The "extreme-terrain held-out" set used in Gate 1's iterative
  domain-randomization testing (§3.4) — a separate, harder terrain config
  for evaluation, not training
- Any change to reward, observation, or termination terms — this is
  terrain-only; the emergent-behavior claim in §3.3.1 depends on the
  reward vector staying as-is

## File layout

```
talon_rl/tasks/locomotion/a1_env/terrain_config/
├── __init__.py          # re-exports A1_ROUGH_TERRAINS_CFG
└── rough_config.py      # the TerrainGeneratorCfg
```

## `rough_config.py`

`A1_ROUGH_TERRAINS_CFG: TerrainGeneratorCfg`, following the same shape as
the vendored reference's `ROUGH_TERRAINS_CFG`
(`size`, `border_width`, `num_rows`, `num_cols`, `horizontal_scale`,
`vertical_scale`, `slope_threshold`, `use_cache=False`, `sub_terrains={...}`),
with `sub_terrains` covering:

- **Climbable obstacles** (proportion > 0): `pyramid_stairs` +
  `pyramid_stairs_inv` (`MeshPyramidStairsTerrainCfg` /
  `MeshInvertedPyramidStairsTerrainCfg`) and `boxes`
  (`MeshRandomGridTerrainCfg`) — same three types the reference config uses
  for this category.
- **Descendable gap/pit** (proportion > 0): `pit` using
  `terrain_gen.MeshPitTerrainCfg` — a pit of configurable depth
  (`pit_depth_range`, meters) with a platform at the center and stairs
  leading back out (`platform_width`, `double_pit: bool` for a two-level
  pit). Confirmed via Isaac Lab's own docs (isaac-sim.github.io/IsaacLab)
  to exist in `isaaclab.terrains` alongside `MeshPyramidStairsTerrainCfg`
  et al. This is the better match for the proposal's "descend slowly vs.
  detour" framing than `MeshGapTerrainCfg` (an impassable-width gap
  *around* a platform — a jump-across obstacle, not a descend-into one).
  **Verification note**: this machine has no Isaac Lab installation to
  actually import against (confirmed via `pip show isaaclab` and a
  filesystem search), so the implementation plan should still treat the
  exact field names (`pit_depth_range` vs. a differently-named equivalent)
  as something to confirm against the installed package at implementation
  time — the class's existence and purpose are confirmed, only the precise
  field spelling isn't.

Proportions are not tuned in this increment (no training has run yet to
inform them) — pick reasonable non-zero values for each of the two
required categories, matching the reference config's rough magnitude
(collectively summing to 1.0 across all `sub_terrains` entries, as
`TerrainGeneratorCfg` requires).

## `a1_env_cfg.py` change

Replace (`a1_env_cfg.py:45`):
```python
ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
```
with a `TerrainImporterCfg` (`isaaclab.terrains.TerrainImporterCfg`) using
`terrain_type="generator"`, `terrain_generator=A1_ROUGH_TERRAINS_CFG`,
`curriculum=False` (per Scope), and the standard fields Isaac Lab's own
velocity-task templates set alongside a generator-backed importer
(`collision_group`, a `physics_material`, `max_init_terrain_level` —
mirror `isaaclab_tasks`' own `cartpole`-adjacent velocity env template or
the vendored reference's `velocity_env_cfg.py` for the exact field set,
verified against the installed `isaaclab` package at implementation time
the same way the rest of this file's imports were verified 2026-09-14 per
its own module docstring).

No other field in `A1SceneCfg` changes. `robot`, `contact_sensor`,
`dome_light` stay as they are.

## Testing

`isaaclab.terrains` requires the real Isaac Lab package to import at all —
same situation as the rest of `a1_env.py`/`a1_env_cfg.py`
(`tests/test_a1_env.py` is GPU/Isaac-Sim-gated via
`pytest.importorskip("isaacsim")` and skips on this repo's default 3.12
`.venv`). Procedural terrain mesh generation is not reasonably mockable
the way `ConstraintManager` was (see
`docs/superpowers/specs/2026-09-15-constraint-manager-vendor-design.md`
for that precedent and why it doesn't apply here) — there is no meaningful
assertion to make about a `TerrainGeneratorCfg`'s mesh output without
actually generating it.

Add one structural assertion to the existing
`test_isaac_lab_env_implements_base_contract` test in `test_a1_env.py`
(same GPU-gated test, not a new file): after `gym.make(...)`, confirm the
scene's terrain was constructed from `A1_ROUGH_TERRAINS_CFG` rather than a
flat ground plane (e.g. checking the scene's terrain importer's
`cfg.terrain_generator is A1_ROUGH_TERRAINS_CFG`, exact attribute path to
be confirmed against the installed `isaaclab.scene`/`isaaclab.terrains` API
during implementation). This only runs on a real GPU machine with Isaac
Sim installed; it cannot be verified in this environment before then.
