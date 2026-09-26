# TienKung Bimanual Box-Carry (Round 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Vendor the TienKung2 Lite asset, define its own MDP config/reward vector, and a physics-free dummy env for MOPPO smoke testing — no real Isaac Lab env yet.

**Architecture:** New sibling module `talon_rl/tasks/manipulation/tienkung_env/` (own `config.py`/`reward.py`/`dummy_env.py`, mirroring `talon_rl/config.py`/`talon_rl/reward.py`/`scripts/moppo/dummy_env.py`'s shapes exactly, own numbers) plus `talon_rl/assets/tienkung2_lite/` (mirrors `talon_rl/assets/unitree_a1/`). A small refactor hoists `TALON_ASSETS_DATA_DIR` and friends out of `unitree_a1/__init__.py` into the shared `talon_rl/assets/__init__.py` (currently empty) so a second robot doesn't duplicate that boilerplate.

**Tech Stack:** Python 3.12, NumPy, gymnasium, pytest (fully runnable in this environment — no Isaac Lab needed except for Task 2's `tienkung.py`, same situation `unitree_a1/a1.py` is already in).

**Spec:** [docs/superpowers/specs/2026-09-15-tienkung-manipulation-design.md](../specs/2026-09-15-tienkung-manipulation-design.md)

## Global Constraints

- `talon_rl/config.py` and `talon_rl/reward.py` (the A1's own MDP definitions, tied to this thesis's `chapter3.tex` tables) are NOT modified — this task's config/reward live in their own new module.
- `RewardVectorCfg.term_names` uses the same 5 names as the A1's (`progress`, `clearance`, `energy`, `impact`, `smoothness`) — same shape, retargeted semantics (see spec's Reward Vector table). No 6th "drop" reward term — box-dropped is a termination, deferred to the real-env increment (out of scope here).
- `PreferenceCfg` is NOT redefined — import `talon_rl.config.PreferenceCfg` directly (already dim-agnostic).
- No command/target-randomization vector this round — the dummy env's implicit goal (a fixed target lift height) is a constructor default, not a per-episode randomized draw.
- The refactor in Task 1 must not change `talon_rl.assets.unitree_a1`'s public behavior — `TALON_A1_CFG`, `TALON_ASSETS_DATA_DIR`, `TALON_ASSETS_METADATA`, `__version__` must all still resolve to the same values as before from the same import paths.
- Task 2 (`tienkung2_lite/tienkung.py`) needs `isaaclab` to import and cannot be tested in this environment — same situation as `unitree_a1/a1.py`, which has no dedicated test file today. Do not attempt to test it; verification happens whenever the real Isaac Lab env is eventually built (out of scope here).
- Tasks 1, 3, 4, 5 have zero `isaaclab` dependency and must be fully unit-tested here with real TDD (RED/GREEN), mirroring `tests/rewards/test_locomotion.py`/`tests/core/envs/test_dummy_env.py`'s existing style exactly.

---

### Task 1: Hoist `TALON_ASSETS_*` into `talon_rl/assets/__init__.py`

**Files:**
- Modify: `talon_rl/assets/__init__.py` (currently empty)
- Modify: `talon_rl/assets/unitree_a1/__init__.py`
- Create: `tests/assets/test_package_init.py`

**Interfaces:**
- Produces: `talon_rl.assets.TALON_ASSETS_EXT_DIR`, `talon_rl.assets.TALON_ASSETS_DATA_DIR`, `talon_rl.assets.TALON_ASSETS_METADATA`, `talon_rl.assets.__version__` — same values as `talon_rl.assets.unitree_a1` exposed before this task, now defined once and re-exported.

- [ ] **Step 1: Write the failing test**

Create `tests/assets/test_package_init.py`:

```python
from pathlib import Path

import talon_rl.assets as assets


def test_talon_assets_data_dir_points_at_assets_data_directory():
    assert assets.TALON_ASSETS_EXT_DIR == Path("talon_rl/assets").resolve()
    assert assets.TALON_ASSETS_DATA_DIR == assets.TALON_ASSETS_EXT_DIR / "data"
    assert assets.TALON_ASSETS_DATA_DIR.is_dir()


def test_talon_assets_metadata_parses_extension_toml():
    assert assets.TALON_ASSETS_METADATA["package"]["version"] == assets.__version__
    assert isinstance(assets.__version__, str)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd talon-rl && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. pytest tests/assets/test_package_init.py -v`
Expected: FAIL — `talon_rl.assets` has no attribute `TALON_ASSETS_EXT_DIR` (the module is currently empty).

- [ ] **Step 3: Write `talon_rl/assets/__init__.py`**

Replace its (currently empty) contents with:

```python
"""Shared path/metadata constants for talon_rl's vendored robot assets.

Hoisted out of unitree_a1/__init__.py (2026-09-15) when a second robot
(tienkung2_lite) was added — the values were always robot-agnostic
(EXT_DIR/DATA_DIR resolve relative to this file's own location, not any
one robot's), only the code that computed them lived inside one robot's
own package. Every per-robot __init__.py re-exports these instead of
re-deriving them — see
docs/superpowers/specs/2026-09-15-tienkung-manipulation-design.md.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

TALON_ASSETS_EXT_DIR = Path(__file__).resolve().parent
"""Path to the assets extension source directory (talon_rl/assets/)."""

TALON_ASSETS_DATA_DIR = TALON_ASSETS_EXT_DIR / "data"
"""Path to the assets data directory (talon_rl/assets/data/)."""

with open(TALON_ASSETS_EXT_DIR / "config" / "extension.toml", "rb") as _f:
    TALON_ASSETS_METADATA = tomllib.load(_f)
"""Extension metadata dictionary parsed from config/extension.toml."""

__version__ = TALON_ASSETS_METADATA["package"]["version"]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd talon-rl && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. pytest tests/assets/test_package_init.py -v`
Expected: PASS.

- [ ] **Step 5: Update `unitree_a1/__init__.py` to re-export instead of re-deriving**

Replace `talon_rl/assets/unitree_a1/__init__.py`'s contents with:

```python
"""Package containing TALON's robot asset configurations.

Mirrors jaykorea/Isaac-RL-Two-wheel-Legged-Bot's assets/<robot>/__init__.py
pattern: per-robot config modules (a1.py) import TALON_ASSETS_DATA_DIR from
the shared talon_rl.assets package instead of each re-deriving it — see
talon_rl/assets/__init__.py.
"""

from __future__ import annotations

from .. import TALON_ASSETS_DATA_DIR  # noqa: F401 — re-exported for a1.py's `from . import TALON_ASSETS_DATA_DIR`

##
# Configuration for different assets.
##

from .a1 import *  # noqa: F401,F403
```

Do NOT edit `talon_rl/assets/unitree_a1/a1.py` — its existing
`from . import TALON_ASSETS_DATA_DIR` keeps resolving correctly because
this `__init__.py` now imports that name into its own namespace before
`from .a1 import *` runs (same execution-order trick the file used before,
just sourcing the name from `..` instead of defining it locally).

- [ ] **Step 6: Verify the existing test suite still passes (regression check)**

Run: `cd talon-rl && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. pytest tests/assets/test_package_init.py tests/rewards/test_locomotion.py tests/core/preferences/test_preferences.py tests/isaaclab/test_constraint_manager.py -v`
Expected: PASS — all tests green, no import errors. (`unitree_a1/__init__.py` still can't be imported here without `isaaclab_assets`, same as before this task — that's unaffected either way since nothing in the runnable suite imports it.)

- [ ] **Step 7: Commit**

```bash
git add talon_rl/assets/__init__.py talon_rl/assets/unitree_a1/__init__.py tests/assets/test_package_init.py
git commit -m "$(cat <<'EOF'
refactor: hoist TALON_ASSETS_DATA_DIR to the shared assets package

Was defined inside unitree_a1/__init__.py even though the value was
always robot-agnostic. Adding tienkung2_lite (next task) is the
trigger to stop duplicating this boilerplate per robot.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Vendor the TienKung2 Lite asset

**Files:**
- Create: `talon_rl/assets/tienkung2_lite/__init__.py`
- Create: `talon_rl/assets/tienkung2_lite/tienkung.py`
- Create: `talon_rl/assets/data/Robots/tienkung2_lite/` (vendored USD files, copied — not authored)

**Interfaces:**
- Consumes: `talon_rl.assets.TALON_ASSETS_DATA_DIR` (Task 1).
- Produces: `talon_rl.assets.tienkung2_lite.tienkung.TALON_TIENKUNG_CFG` — an `isaaclab.assets.articulation.ArticulationCfg`.

This task cannot be tested in this environment (`isaaclab` not installed) — no RED/GREEN cycle. Follow the steps as written; this is a transcription + file-copy task.

- [ ] **Step 1: Copy the vendored USD files**

Copy the entire `usd/` directory from the source project into
`talon_rl/assets/data/Robots/tienkung2_lite/`:

```bash
mkdir -p "talon_rl/assets/data/Robots/tienkung2_lite"
cp -r "/home/xero/Master's Degree/Thesis/TienKung-Lab-main/legged_lab/assets/tienkung2_lite/usd/." \
      "talon_rl/assets/data/Robots/tienkung2_lite/"
```

This vendors `tienkung2_lite.usd`, `config.yaml`, `.asset_hash`, and the
`configuration/` subdirectory (`tienkung2_lite_base.usd`,
`tienkung2_lite_physics.usd`, `tienkung2_lite_sensor.usd`) — matching the
scope of what `unitree_a1`'s own vendoring copied (final USD assets, not
the intermediate URDF/MJCF/mesh sources also present in the source
project's `tienkung2_lite/` directory).

- [ ] **Step 2: Create the package `__init__.py`**

Create `talon_rl/assets/tienkung2_lite/__init__.py`:

```python
"""Package containing TienKung2 Lite's asset configuration.

Separate research application (bimanual box-carry) reusing this repo's
generic Multi-Objective RMA infrastructure, not part of this thesis's own
A1 locomotion scope — see
docs/superpowers/specs/2026-09-15-tienkung-manipulation-design.md.
"""

from __future__ import annotations

from .. import TALON_ASSETS_DATA_DIR  # noqa: F401 — re-exported for tienkung.py's `from . import TALON_ASSETS_DATA_DIR`

from .tienkung import *  # noqa: F401,F403
```

- [ ] **Step 3: Write `tienkung.py`**

Create `talon_rl/assets/tienkung2_lite/tienkung.py`. This vendors
`TIENKUNG2LITE_CFG` from
`TienKung-Lab-main/legged_lab/assets/tienkung2_lite/tienkung.py` verbatim
(joint positions, actuator groups unchanged from the source — no
equivalent of the A1's RMA-paper Kp/Kd retuning exists here yet), renamed
to this repo's `TALON_*` naming convention and pointed at the vendored
local USD path instead of the source project's `ISAAC_ASSET_DIR`:

```python
# talon_rl/assets/tienkung2_lite/tienkung.py
"""Talon's TienKung2 Lite config — vendors TIENKUNG2LITE_CFG from
TienKung-Lab (legged_lab/assets/tienkung2_lite/tienkung.py) verbatim
(joint positions, actuator gains unchanged from the source), overriding
usd_path to a locally vendored copy. Separate research application
(bimanual box-carry, not this thesis's own A1 locomotion) that reuses this
repo's generic Multi-Objective RMA infrastructure — see
docs/superpowers/specs/2026-09-15-tienkung-manipulation-design.md.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from . import TALON_ASSETS_DATA_DIR

TALON_TIENKUNG_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{TALON_ASSETS_DATA_DIR}/Robots/tienkung2_lite/tienkung2_lite.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        joint_pos={
            "hip_roll_l_joint": 0.0,
            "hip_pitch_l_joint": -0.5,
            "hip_yaw_l_joint": 0.0,
            "knee_pitch_l_joint": 1.0,
            "ankle_pitch_l_joint": -0.5,
            "ankle_roll_l_joint": -0.0,
            "hip_roll_r_joint": -0.0,
            "hip_pitch_r_joint": -0.5,
            "hip_yaw_r_joint": 0.0,
            "knee_pitch_r_joint": 1.0,
            "ankle_pitch_r_joint": -0.5,
            "ankle_roll_r_joint": 0.0,
            "shoulder_pitch_l_joint": 0.0,
            "shoulder_roll_l_joint": 0.1,
            "shoulder_yaw_l_joint": -0.0,
            "elbow_pitch_l_joint": -0.3,
            "shoulder_pitch_r_joint": 0.0,
            "shoulder_roll_r_joint": -0.1,
            "shoulder_yaw_r_joint": 0.0,
            "elbow_pitch_r_joint": -0.3,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "hip_roll_.*_joint",
                "hip_pitch_.*_joint",
                "hip_yaw_.*_joint",
                "knee_pitch_.*_joint",
            ],
            effort_limit_sim={
                "hip_roll_.*_joint": 180,
                "hip_pitch_.*_joint": 300,
                "hip_yaw_.*_joint": 180,
                "knee_pitch_.*_joint": 300,
            },
            velocity_limit_sim={
                "hip_roll_.*_joint": 15.6,
                "hip_pitch_.*_joint": 15.6,
                "hip_yaw_.*_joint": 15.6,
                "knee_pitch_.*_joint": 15.6,
            },
            stiffness={
                "hip_roll_.*_joint": 700,
                "hip_pitch_.*_joint": 700,
                "hip_yaw_.*_joint": 500,
                "knee_pitch_.*_joint": 700,
            },
            damping={
                "hip_roll_.*_joint": 10,
                "hip_pitch_.*_joint": 10,
                "hip_yaw_.*_joint": 5,
                "knee_pitch_.*_joint": 10,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[
                "ankle_pitch_.*_joint",
                "ankle_roll_.*_joint",
            ],
            effort_limit_sim={
                "ankle_pitch_.*_joint": 60,
                "ankle_roll_.*_joint": 30,
            },
            velocity_limit_sim={
                "ankle_pitch_.*_joint": 12.8,
                "ankle_roll_.*_joint": 7.8,
            },
            stiffness={
                "ankle_pitch_.*_joint": 30,
                "ankle_roll_.*_joint": 16.8,
            },
            damping={
                "ankle_pitch_.*_joint": 2.5,
                "ankle_roll_.*_joint": 1.4,
            },
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulder_pitch_.*_joint",
                "shoulder_roll_.*_joint",
                "shoulder_yaw_.*_joint",
                "elbow_pitch_.*_joint",
            ],
            effort_limit_sim={
                "shoulder_pitch_.*_joint": 52.5,
                "shoulder_roll_.*_joint": 52.5,
                "shoulder_yaw_.*_joint": 52.5,
                "elbow_pitch_.*_joint": 52.5,
            },
            velocity_limit_sim={
                "shoulder_pitch_.*_joint": 14.1,
                "shoulder_roll_.*_joint": 14.1,
                "shoulder_yaw_.*_joint": 14.1,
                "elbow_pitch_.*_joint": 14.1,
            },
            stiffness={
                "shoulder_pitch_.*_joint": 60,
                "shoulder_roll_.*_joint": 20,
                "shoulder_yaw_.*_joint": 10,
                "elbow_pitch_.*_joint": 10,
            },
            damping={
                "shoulder_pitch_.*_joint": 3,
                "shoulder_roll_.*_joint": 1.5,
                "shoulder_yaw_.*_joint": 1,
                "elbow_pitch_.*_joint": 1,
            },
        ),
    },
)
```

- [ ] **Step 4: Commit**

```bash
git add talon_rl/assets/tienkung2_lite talon_rl/assets/data/Robots/tienkung2_lite
git commit -m "$(cat <<'EOF'
feat: vendor TienKung2 Lite asset

Verbatim port of TIENKUNG2LITE_CFG from TienKung-Lab, renamed to this
repo's TALON_* convention with usd_path pointed at a locally vendored
copy — mirrors the unitree_a1 asset-vendoring pattern exactly. No
actuator retuning this round. Untested in this environment (no
isaaclab installed here), same situation unitree_a1/a1.py is already
in — verification happens when a real Isaac Lab env is built for this
task (a later increment, out of scope here).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: `tienkung_env/config.py`

**Files:**
- Create: `talon_rl/tasks/manipulation/__init__.py`
- Create: `talon_rl/tasks/manipulation/tienkung_env/__init__.py`
- Create: `talon_rl/tasks/manipulation/tienkung_env/config.py`
- Create: `tests/tasks/tienkung/test_config.py`

**Interfaces:**
- Produces: `talon_rl.tasks.manipulation.tienkung_env.config.ObservationSpaceCfg` (fields: `joint_pos_dim=8`, `joint_vel_dim=8`, `box_relative_pos_dim=3`, `arm_contact_dim=2`, `prev_action_dim=8`, `preference_dim=5`, property `total_dim`), `ActionSpaceCfg` (field `dim=8`), `RewardVectorCfg` (fields `term_names=("progress","clearance","energy","impact","smoothness")`, `active=(True,)*5`, `progress_std=0.3`, property `dim`).

- [ ] **Step 1: Write the failing test**

Create `tests/tasks/tienkung/test_config.py`:

```python
from talon_rl.tasks.manipulation.tienkung_env.config import (
    ActionSpaceCfg,
    ObservationSpaceCfg,
    RewardVectorCfg,
)


def test_observation_space_total_dim_sums_all_fields():
    cfg = ObservationSpaceCfg()
    assert cfg.total_dim == (
        cfg.joint_pos_dim
        + cfg.joint_vel_dim
        + cfg.box_relative_pos_dim
        + cfg.arm_contact_dim
        + cfg.prev_action_dim
        + cfg.preference_dim
    )
    assert cfg.total_dim == 34


def test_action_space_dim_matches_arm_dof():
    assert ActionSpaceCfg().dim == 8


def test_reward_vector_cfg_dim_matches_term_count():
    cfg = RewardVectorCfg()
    assert cfg.term_names == ("progress", "clearance", "energy", "impact", "smoothness")
    assert cfg.dim == 5
    assert len(cfg.active) == 5
    assert all(cfg.active)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd talon-rl && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. pytest tests/tasks/tienkung/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'talon_rl.tasks.manipulation'`

- [ ] **Step 3: Create the package `__init__.py` files**

Create `talon_rl/tasks/manipulation/__init__.py`:

```python
"""Task package for manipulation embodiments — separate from
talon_rl/tasks/locomotion/ (this thesis's own A1 scope). See
docs/superpowers/specs/2026-09-15-tienkung-manipulation-design.md.
"""
```

Create `talon_rl/tasks/manipulation/tienkung_env/__init__.py`:

```python
"""TienKung2 Lite bimanual box-carry task."""
```

- [ ] **Step 4: Write `config.py`**

Create `talon_rl/tasks/manipulation/tienkung_env/config.py`:

```python
# talon_rl/tasks/manipulation/tienkung_env/config.py
"""MDP configuration dataclasses for TienKung's bimanual box-carry task.

Same shape as talon_rl/config.py's ObservationSpaceCfg/ActionSpaceCfg/
RewardVectorCfg (this thesis's own A1 locomotion config), own numbers —
deliberately NOT the same file, since talon_rl/config.py mirrors this
thesis's own chapter3.tex tables specifically. PreferenceCfg is not
redefined here: talon_rl.config.PreferenceCfg is already dim-agnostic
(operates on reward_cfg.dim), so this task imports that one directly. See
docs/superpowers/specs/2026-09-15-tienkung-manipulation-design.md.

Prelim-scope note (mirrors talon_rl/config.py's own): no adaptation-module-
derived latent is part of this observation — the box's mass/fragility
isn't observable here. Inferring it online via the same Adaptation Module
mechanism this thesis already defines for the A1's payload mass is a later
increment.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ObservationSpaceCfg:
    """Prelim observation layout. Total dim = sum of the fields below."""

    joint_pos_dim: int = 8  # 4 joint types (shoulder_pitch/roll/yaw, elbow_pitch) x 2 arms
    joint_vel_dim: int = 8
    box_relative_pos_dim: int = 3  # box position relative to torso frame
    arm_contact_dim: int = 2  # binarized L/R arm-box contact
    prev_action_dim: int = 8
    preference_dim: int = 5  # w — one weight per reward-vector term

    @property
    def total_dim(self) -> int:
        return (
            self.joint_pos_dim
            + self.joint_vel_dim
            + self.box_relative_pos_dim
            + self.arm_contact_dim
            + self.prev_action_dim
            + self.preference_dim
        )


@dataclass(frozen=True)
class ActionSpaceCfg:
    """a_t: target joint angle per arm DOF, PD-converted to torque downstream."""

    dim: int = 8


@dataclass(frozen=True)
class RewardVectorCfg:
    """Same 5 term names as talon_rl.config.RewardVectorCfg (the A1's) —
    same multi-objective shape, retargeted semantics (see the design spec's
    Reward Vector table). `active` mirrors that file's pattern: kept here,
    not hardcoded, so a later round can add terms without touching
    preference/MOPPO code.
    """

    term_names: tuple[str, ...] = ("progress", "clearance", "energy", "impact", "smoothness")
    active: tuple[bool, ...] = field(default_factory=lambda: (True, True, True, True, True))

    # Reward-shaping constants — rough starting points, not tuned (no
    # training has run yet), same posture as talon_rl/config.py's own.
    progress_std: float = 0.3  # exp-kernel std for box-height tracking (meters)

    @property
    def dim(self) -> int:
        return len(self.term_names)
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `cd talon-rl && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. pytest tests/tasks/tienkung/test_config.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add talon_rl/tasks/manipulation tests/tasks/tienkung/test_config.py
git commit -m "$(cat <<'EOF'
feat: add TienKung MDP config dataclasses

Same shape as talon_rl/config.py's ObservationSpaceCfg/ActionSpaceCfg/
RewardVectorCfg, own numbers for the 8-DOF arms-only bimanual
box-carry task. Reuses talon_rl.config.PreferenceCfg directly (already
dim-agnostic) rather than redefining it.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: `tienkung_env/reward.py`

**Files:**
- Create: `talon_rl/tasks/manipulation/tienkung_env/reward.py`
- Create: `tests/tasks/tienkung/test_reward.py`

**Interfaces:**
- Consumes: `talon_rl.tasks.manipulation.tienkung_env.config.RewardVectorCfg` (Task 3).
- Produces: `progress_reward(box_height, target_height, std) -> (N,)`, `clearance_reward(obstacle_dist, safe_dist=0.5) -> (N,)`, `energy_reward(joint_torque, joint_vel) -> (N,)`, `impact_reward(arm_contact_force, threshold=50.0) -> (N,)`, `smoothness_reward(action, prev_action, joint_acc) -> (N,)`, `compute_reward_vector(transition, cfg) -> (N, dim)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/tasks/tienkung/test_reward.py`:

```python
import numpy as np

from talon_rl.tasks.manipulation.tienkung_env.config import RewardVectorCfg
from talon_rl.tasks.manipulation.tienkung_env.reward import (
    clearance_reward,
    compute_reward_vector,
    energy_reward,
    impact_reward,
    progress_reward,
    smoothness_reward,
)


def test_progress_reward_is_max_at_zero_error():
    height = np.array([1.0, 1.0, 1.0])
    target = np.array([1.0, 1.0, 1.0])
    r_same = progress_reward(height, target, std=0.3)
    r_diff = progress_reward(height, target * 2.0, std=0.3)
    assert r_same.shape == (3,)
    assert np.allclose(r_same, 1.0)
    assert np.all(r_diff < 1.0)


def test_clearance_reward_clips_to_unit_interval():
    dist = np.array([10.0, 0.0])
    r = clearance_reward(dist, safe_dist=0.5)
    assert r.shape == (2,)
    assert r[0] == 1.0
    assert r[1] == 0.0


def test_energy_reward_is_nonpositive():
    torque = np.array([[1.0, -2.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0] * 8])
    vel = np.ones((2, 8))
    r = energy_reward(torque, vel)
    assert r.shape == (2,)
    assert r[0] <= 0.0
    assert r[1] == 0.0


def test_impact_reward_penalizes_only_above_threshold():
    forces = np.array([[10.0, 10.0], [200.0, 0.0]])
    r = impact_reward(forces, threshold=50.0)
    assert r.shape == (2,)
    assert r[0] == 0.0
    assert r[1] < 0.0


def test_smoothness_reward_penalizes_action_change():
    a = np.zeros((2, 8))
    b = np.ones((2, 8))
    acc = np.zeros((2, 8))
    r_same = smoothness_reward(a, a, acc)
    r_diff = smoothness_reward(b, a, acc)
    assert r_same.shape == (2,)
    assert np.allclose(r_same, 0.0)
    assert np.all(r_diff < 0.0)


def test_compute_reward_vector_respects_active_mask_and_order():
    cfg = RewardVectorCfg(active=(True, False, True, False, True))
    n = 4
    transition = {
        "obs": np.zeros((n, 1)),
        "box_height": np.ones(n), "target_height": np.ones(n),
        "obstacle_dist": np.ones(n),
        "joint_torque": np.ones((n, 8)), "joint_vel": np.ones((n, 8)),
        "arm_contact_force": np.zeros((n, 2)),
        "action": np.zeros((n, 8)), "prev_action": np.zeros((n, 8)), "joint_acc": np.zeros((n, 8)),
    }
    r = compute_reward_vector(transition, cfg)
    assert r.shape == (n, 5)
    assert np.allclose(r[:, 1], 0.0)  # clearance masked off
    assert np.allclose(r[:, 3], 0.0)  # impact masked off
    assert np.all(r[:, 0] != 0.0)     # progress active
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd talon-rl && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. pytest tests/tasks/tienkung/test_reward.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'talon_rl.tasks.manipulation.tienkung_env.reward'`

- [ ] **Step 3: Write `reward.py`**

Create `talon_rl/tasks/manipulation/tienkung_env/reward.py`:

```python
# talon_rl/tasks/manipulation/tienkung_env/reward.py
"""The 5-term reward vector for TienKung's bimanual box-carry task —
same shape as talon_rl/reward.py's (this thesis's own A1 locomotion reward
vector), retargeted semantics: box-height tracking instead of velocity
tracking, arm-contact force instead of foot-contact force, 8 arm DOF
instead of 12 leg DOF. See
docs/superpowers/specs/2026-09-15-tienkung-manipulation-design.md's Reward
Vector table for the full A1-analog mapping.

Batched (N, ...) -> (N,) throughout, same convention as talon_rl/reward.py.
None of these terms know about the preference vector w — same as that
file, arbitration happens downstream in the MOPPO training loop.
"""

from __future__ import annotations

import numpy as np

from .config import RewardVectorCfg


def progress_reward(box_height: np.ndarray, target_height: np.ndarray, std: float) -> np.ndarray:
    """Exp-kernel tracking of box height toward the lift-and-hold target.
    (N,), (N,) -> (N,)."""
    err = (box_height - target_height) ** 2
    return np.exp(-err / (std**2)).astype(np.float32)


def clearance_reward(obstacle_dist: np.ndarray, safe_dist: float = 0.5) -> np.ndarray:
    """Distance to obstacles while reaching for the box — scripted signal
    until a real exteroception source exists (out of scope here), same
    formula as talon_rl.reward.clearance_reward. (N,) -> (N,)."""
    return np.clip(obstacle_dist / safe_dist, 0.0, 1.0).astype(np.float32)


def energy_reward(joint_torque: np.ndarray, joint_vel: np.ndarray) -> np.ndarray:
    """Raw arm power per step, negated. (N, 8), (N, 8) -> (N,)."""
    power = np.sum(np.abs(joint_torque * joint_vel), axis=-1)
    return (-power).astype(np.float32)


def impact_reward(arm_contact_force: np.ndarray, threshold: float = 50.0) -> np.ndarray:
    """Continuous grip-force mitigation — a privileged, sim-only training
    signal (no real force/torque sensor exists on this hardware; reward
    computation runs at training time only, so this never needs to be
    sim-to-real transferable). Threshold is an arbitrary prelim
    placeholder. (N, 2) -> (N,)."""
    if arm_contact_force.shape[-1] == 0:
        return np.zeros(arm_contact_force.shape[0], dtype=np.float32)
    peak = np.max(np.abs(arm_contact_force), axis=-1)
    return (-np.maximum(0.0, peak - threshold) / threshold).astype(np.float32)


def smoothness_reward(action: np.ndarray, prev_action: np.ndarray, joint_acc: np.ndarray) -> np.ndarray:
    """Fixed-weight regularizer, same formula as talon_rl.reward's, over 8
    arm DOF instead of 12. (N, 8) each -> (N,)."""
    action_rate = np.sum((action - prev_action) ** 2, axis=-1)
    acc = np.sum(joint_acc**2, axis=-1)
    return (-(action_rate + 0.01 * acc)).astype(np.float32)


_TERM_FUNCS = {
    "progress": lambda t, cfg: progress_reward(t["box_height"], t["target_height"], cfg.progress_std),
    "clearance": lambda t, cfg: clearance_reward(t["obstacle_dist"]),
    "energy": lambda t, cfg: energy_reward(t["joint_torque"], t["joint_vel"]),
    "impact": lambda t, cfg: impact_reward(t["arm_contact_force"]),
    "smoothness": lambda t, cfg: smoothness_reward(t["action"], t["prev_action"], t["joint_acc"]),
}


def compute_reward_vector(transition: dict, cfg: RewardVectorCfg) -> np.ndarray:
    """Returns r as an (N, dim) array in cfg.term_names order. Inactive terms are 0."""
    n = transition["obs"].shape[0]
    values = []
    for name, active in zip(cfg.term_names, cfg.active):
        values.append(_TERM_FUNCS[name](transition, cfg) if active else np.zeros(n, dtype=np.float32))
    return np.stack(values, axis=-1).astype(np.float32)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd talon-rl && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. pytest tests/tasks/tienkung/test_reward.py -v`
Expected: PASS — all 6 tests.

- [ ] **Step 5: Commit**

```bash
git add talon_rl/tasks/manipulation/tienkung_env/reward.py tests/tasks/tienkung/test_reward.py
git commit -m "$(cat <<'EOF'
feat: add TienKung 5-term reward vector

Same shape as talon_rl/reward.py's A1 reward vector, retargeted:
box-height tracking, arm-contact force (privileged sim-only signal,
no real F/T sensor on this hardware), 8 arm DOF. Box-dropped is
deliberately NOT a 6th reward term here — mirrors the A1's own
convention of handling catastrophic failure as a termination, deferred
to the real-env increment.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: `tienkung_env/dummy_env.py`

**Files:**
- Create: `talon_rl/tasks/manipulation/tienkung_env/dummy_env.py`
- Create: `tests/tasks/tienkung/test_dummy_env.py`

**Interfaces:**
- Consumes: `talon_rl.tasks.manipulation.tienkung_env.config.{ObservationSpaceCfg, ActionSpaceCfg}` (Task 3), `talon_rl.envs.base_env.BaseTalonEnv` (existing).
- Produces: `DummyTalonEnv(obs_cfg, action_cfg, num_envs=1, horizon=200, dt=0.02, seed=0, target_height=1.0)` implementing `BaseTalonEnv`; transition dict keys `box_height`, `target_height`, `obstacle_dist`, `joint_torque`, `joint_vel`, `joint_acc`, `arm_contact_force`, `action`, `prev_action` (matches Task 4's `reward.py._TERM_FUNCS` requirements exactly).

- [ ] **Step 1: Write the failing tests**

Create `tests/tasks/tienkung/test_dummy_env.py`:

```python
import numpy as np

from talon_rl.envs.base_env import BaseTalonEnv
from talon_rl.tasks.manipulation.tienkung_env.config import ActionSpaceCfg, ObservationSpaceCfg
from talon_rl.tasks.manipulation.tienkung_env.dummy_env import DummyTalonEnv


def _make_env(num_envs=4, horizon=5, seed=0):
    obs_cfg = ObservationSpaceCfg()
    action_cfg = ActionSpaceCfg()
    env = DummyTalonEnv(obs_cfg, action_cfg, num_envs=num_envs, horizon=horizon, seed=seed)
    return env, obs_cfg, action_cfg


def test_satisfies_base_contract_and_shapes():
    env, obs_cfg, action_cfg = _make_env(num_envs=4)
    assert isinstance(env, BaseTalonEnv)
    assert env.num_envs == 4
    assert env.obs_dim == obs_cfg.total_dim - obs_cfg.preference_dim
    assert env.action_dim == action_cfg.dim

    transition = env.reset()
    assert transition["obs"].shape == (4, env.obs_dim)
    for key in (
        "box_height", "target_height", "obstacle_dist", "joint_torque", "joint_vel",
        "joint_acc", "arm_contact_force", "action", "prev_action",
    ):
        assert key in transition
        assert transition[key].shape[0] == 4

    action = np.zeros((4, env.action_dim), dtype=np.float32)
    transition, done = env.step(action)
    assert transition["obs"].shape == (4, env.obs_dim)
    assert done.shape == (4,)
    assert done.dtype == bool


def test_lanes_auto_reset_independently():
    """horizon=1 forces every lane to truncate every single step — after
    enough steps, each lane must have resampled its own obstacle_dist
    (proves auto-reset runs per-lane, not just once globally)."""
    env, _, action_cfg = _make_env(num_envs=8, horizon=1, seed=0)
    env.reset()
    action = np.zeros((8, action_cfg.dim), dtype=np.float32)

    obstacle_dists_seen = set()
    for _ in range(10):
        transition, done = env.step(action)
        assert np.all(done)  # horizon=1: every lane truncates every step
        for val in transition["obstacle_dist"]:
            obstacle_dists_seen.add(round(float(val), 4))

    assert len(obstacle_dists_seen) > 5


def test_smoothness_uses_correct_prev_action_ordering():
    """joint_acc = (action - prev_action) / dt must use the PREVIOUS
    step's action, not this step's — same regression concern the A1's
    DummyEnv guards against."""
    env, _, action_cfg = _make_env(num_envs=1, horizon=200, seed=0)
    env.reset()

    a1 = np.full((1, action_cfg.dim), 0.5, dtype=np.float32)
    t1, _ = env.step(a1)
    assert np.allclose(t1["prev_action"], 0.0)
    assert np.allclose(t1["action"], 0.5)

    a2 = np.full((1, action_cfg.dim), -0.3, dtype=np.float32)
    t2, _ = env.step(a2)
    assert np.allclose(t2["prev_action"], 0.5)
    assert np.allclose(t2["action"], -0.3)


def test_positive_action_0_raises_box_height_toward_target():
    env, _, action_cfg = _make_env(num_envs=1, horizon=200, seed=0)
    t0 = env.reset()
    action = np.zeros((1, action_cfg.dim), dtype=np.float32)
    action[0, 0] = 1.0  # max lift drive
    t1, _ = env.step(action)
    assert t1["box_height"][0] > t0["box_height"][0]


def test_action_1_drives_arm_contact_force():
    env, _, action_cfg = _make_env(num_envs=1, horizon=200, seed=0)
    env.reset()
    gentle = np.zeros((1, action_cfg.dim), dtype=np.float32)
    firm = np.zeros((1, action_cfg.dim), dtype=np.float32)
    firm[0, 1] = 1.0
    t_gentle, _ = env.step(gentle)
    env.reset()
    t_firm, _ = env.step(firm)
    assert np.all(t_firm["arm_contact_force"] > t_gentle["arm_contact_force"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd talon-rl && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. pytest tests/tasks/tienkung/test_dummy_env.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'talon_rl.tasks.manipulation.tienkung_env.dummy_env'`

- [ ] **Step 3: Write `dummy_env.py`**

Create `talon_rl/tasks/manipulation/tienkung_env/dummy_env.py`:

```python
# talon_rl/tasks/manipulation/tienkung_env/dummy_env.py
"""Physics-free smoke-test env for TienKung's bimanual box-carry —
same batch-native, gym.vector.SyncVectorEnv-backed structure as
scripts/moppo/dummy_env.py's DummyEnv/DummyTalonEnv (this thesis's own A1
dummy env), own toy physics: action[0] drives box-lift progress,
action[1] drives grip firmness (higher = firmer hold = more arm-contact
force / impact risk, lower = gentler but risks the box slipping) —
retargets that file's action[0]=accel/action[1]=softness trade-off
pattern from locomotion to bimanual carrying.

No termination condition beyond horizon this round (box-dropped
termination logic is deferred to the real Isaac Lab env, a later
increment) — every episode truncates at `horizon` steps, never
terminates early.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from talon_rl.envs.base_env import BaseTalonEnv

from .config import ActionSpaceCfg, ObservationSpaceCfg


class DummyEnv(gym.Env):
    """Single lane. action[0] drives box-lift height toward target_height;
    action[1] drives grip firmness (arm_contact_force magnitude)."""

    def __init__(
        self,
        obs_cfg: ObservationSpaceCfg,
        action_cfg: ActionSpaceCfg,
        horizon: int = 200,
        dt: float = 0.02,
        target_height: float = 1.0,
    ):
        super().__init__()
        self.obs_cfg = obs_cfg
        self.action_cfg = action_cfg
        self.horizon = horizon
        self.dt = dt
        self.target_height = target_height
        self.obs_dim = obs_cfg.total_dim - obs_cfg.preference_dim
        self.action_dim = action_cfg.dim
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.action_dim,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.height = 0.0
        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self._obstacle_dist = float(self.np_random.uniform(3.0, 6.0))
        obs = self._compute_transition(np.zeros(self.action_dim, dtype=np.float32))
        return obs, {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        lift_rate = max(0.0, float(action[0])) * 0.5
        self.height = float(np.clip(self.height + lift_rate * self.dt, 0.0, 2.0))
        self.t += 1

        obs = self._compute_transition(action)
        self.prev_action = action  # advance AFTER _compute_transition reads the old value

        terminated = False  # no drop condition this round — see module docstring
        truncated = bool(self.t >= self.horizon)
        return obs, 0.0, terminated, truncated, {}

    def _compute_transition(self, action: np.ndarray) -> np.ndarray:
        """Computes and stores this step's transition fields (read
        externally by DummyTalonEnv after step()/reset() return) using
        self.prev_action's CURRENT (pre-advance) value — must run before
        step() advances self.prev_action to `action`, matching the A1
        DummyEnv's own ordering rule."""
        firmness = float(np.clip(action[1], 0.0, 1.0)) if action.size > 1 else 0.0

        self.box_height = np.float32(self.height)
        self.target_height_out = np.float32(self.target_height)
        self.obstacle_dist = self._obstacle_dist
        self.joint_vel = np.full(self.action_dim, self.height, dtype=np.float32)
        self.joint_torque = action * (1.0 + firmness)
        self.joint_acc = (action - self.prev_action) / self.dt
        self.arm_contact_force = np.full(2, firmness * 60.0, dtype=np.float32)
        self.action = action
        self.prev_action_out = self.prev_action

        obs = np.concatenate([
            np.zeros(self.obs_cfg.joint_pos_dim, dtype=np.float32),
            np.full(self.obs_cfg.joint_vel_dim, self.height, dtype=np.float32),
            np.array([0.0, 0.0, self.height], dtype=np.float32)[: self.obs_cfg.box_relative_pos_dim],
            np.ones(self.obs_cfg.arm_contact_dim, dtype=np.float32),
            self.prev_action[: self.obs_cfg.prev_action_dim],
        ])
        return obs.astype(np.float32)


class DummyTalonEnv(BaseTalonEnv):
    def __init__(
        self,
        obs_cfg: ObservationSpaceCfg,
        action_cfg: ActionSpaceCfg,
        num_envs: int = 1,
        horizon: int = 200,
        dt: float = 0.02,
        seed: int = 0,
        target_height: float = 1.0,
    ):
        self.num_envs = num_envs
        self.obs_dim = obs_cfg.total_dim - obs_cfg.preference_dim
        self.action_dim = action_cfg.dim
        self._seed = seed
        self._vec_env = gym.vector.SyncVectorEnv(
            [
                (lambda: DummyEnv(obs_cfg, action_cfg, horizon=horizon, dt=dt, target_height=target_height))
                for _ in range(num_envs)
            ],
            autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
        )

    def reset(self) -> dict:
        obs, _infos = self._vec_env.reset(seed=[self._seed + i for i in range(self.num_envs)])
        return self._collect_transition(obs)

    def step(self, action: np.ndarray) -> tuple[dict, np.ndarray]:
        obs, _rewards, terminated, truncated, _infos = self._vec_env.step(action)
        done = np.logical_or(terminated, truncated)
        return self._collect_transition(obs), done

    def _collect_transition(self, obs: np.ndarray) -> dict:
        transition = {"obs": obs.astype(np.float32)}
        for key, attr in (
            ("box_height", "box_height"),
            ("target_height", "target_height_out"),
            ("obstacle_dist", "obstacle_dist"),
            ("joint_torque", "joint_torque"),
            ("joint_vel", "joint_vel"),
            ("joint_acc", "joint_acc"),
            ("arm_contact_force", "arm_contact_force"),
            ("action", "action"),
        ):
            transition[key] = np.stack([getattr(env.unwrapped, attr) for env in self._vec_env.envs])
        transition["prev_action"] = np.stack([env.unwrapped.prev_action_out for env in self._vec_env.envs])
        return transition

    def close(self) -> None:
        self._vec_env.close()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd talon-rl && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. pytest tests/tasks/tienkung/test_dummy_env.py -v`
Expected: PASS — all 5 tests.

- [ ] **Step 5: Run the full new+existing suite to confirm no regressions**

Run: `cd talon-rl && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. pytest tests/assets/test_package_init.py tests/rewards/test_locomotion.py tests/core/preferences/test_preferences.py tests/isaaclab/test_constraint_manager.py tests/tasks/tienkung/test_config.py tests/tasks/tienkung/test_reward.py tests/tasks/tienkung/test_dummy_env.py -v`
Expected: PASS — 2 (assets_init) + 6 (reward) + 5 (preference) + 7 (constraint_manager) + 3 (tienkung_config) + 6 (tienkung_reward) + 5 (tienkung_dummy_env) = 34 passed.

- [ ] **Step 6: Commit**

```bash
git add talon_rl/tasks/manipulation/tienkung_env/dummy_env.py tests/tasks/tienkung/test_dummy_env.py
git commit -m "$(cat <<'EOF'
feat: add TienKung physics-free dummy env

Same batch-native, SyncVectorEnv-backed structure as this thesis's own
A1 dummy env, own toy physics for the bimanual box-carry task:
action[0] drives lift progress, action[1] drives grip firmness (arm
contact force). Completes round 1 (asset + config + reward + dummy
env) of the TienKung manipulation module — no real Isaac Lab env yet.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```
