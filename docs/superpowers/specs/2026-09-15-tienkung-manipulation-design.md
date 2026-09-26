# TienKung bimanual box-carry — asset, config, reward, dummy env (design)

## Context

Separate research application applying this thesis's Multi-Objective RMA
principles to a different embodiment (TienKung2 Lite, a bipedal humanoid,
`TienKung-Lab-main/legged_lab/assets/tienkung2_lite`) and a different task
(bimanual box carrying, not locomotion). Explicitly out of this thesis's own
defended scope (`00_Proposal §3.6` disclaims cross-embodiment transfer
across different joint topologies), but the user has chosen to keep it
inside talon-rl as a sibling module rather than a separate repo, since it
reuses this repo's generic infrastructure directly.

TienKung's hardware has no gripper and no force/torque sensor — the task is
two arms wrapping around a box from both sides (a "hug," not a pinch grasp),
matching the reference asset's actuator groups (`legs`, `feet`, `arms` —
`ImplicitActuatorCfg`, no hand/gripper actuator, no contact/F-T sensor
class). This round is **arms-only** (8 DOF: `shoulder_pitch/roll/yaw`,
`elbow_pitch` × L/R) — the robot's base is held fixed, no locomotion. Legs
and feet are out of scope for this task.

Because the box's mass/fragility isn't known in advance ("หยิบ object
อะไรก็ได้"), this maps directly onto Rapid Motor Adaptation, the same
mechanism this thesis already uses for the A1's payload-mass Adaptation
Module (`00_Proposal §3.2.1`) — reused conceptually, not reimplemented here
(see Out of Scope).

## Scope

**In scope this round**:
- Vendor the `tienkung2_lite` asset into `talon_rl/assets/`, mirroring the
  existing `unitree_a1` pattern exactly.
- A TienKung-specific `ObservationSpaceCfg`/`ActionSpaceCfg`/
  `RewardVectorCfg` (own module, not touching `talon_rl/config.py` — that
  file mirrors *this thesis's own* `chapter3.tex` tables and is
  A1-specific).
- `reward.py` with 5 terms (see Reward Vector below) — deliberately the
  same term *count* and *shape* as the A1's, retargeted semantics.
- A physics-free dummy env (mirrors `scripts/moppo/dummy_env.py`'s
  batch-native, `gym.vector.SyncVectorEnv`-backed structure, own toy
  physics) for early MOPPO smoke testing.
- The `TALON_ASSETS_DATA_DIR`/`TALON_ASSETS_EXT_DIR`/metadata-parsing
  refactor described under Changes to existing code.

**Out of scope this round** (later increments):
- The real Isaac Lab env (scene/observations/actions/terminations wired to
  Isaac Sim) — same incremental path the A1 env itself took across several
  prior plans (asset → config/reward → dummy env → real env).
- The Adaptation Module itself (teacher-privileged training + distillation
  to a proprioception-only student) — this thesis's own A1 Adaptation
  Module is *also* not built yet (`talon_rl/config.py`'s own docstring:
  "the Adaptation Module ... [is] NOT part of this prelim's observation ...
  marked [TBD]"). TienKung's manipulation task inherits the same
  prelim-scope boundary — this round's observation space has no
  adaptation-derived latent either.
- Any locomotion/base movement — arms-only, base fixed.
- Target-pose randomization / a "command" vector — this round's implicit
  goal is fixed (lift the box off the surface and hold it stably against
  the torso), not a randomized target like the A1's velocity command.

## File layout

```
talon_rl/assets/
├── __init__.py                    MODIFIED — hoists TALON_ASSETS_* here
├── unitree_a1/__init__.py         MODIFIED — imports the hoisted constants
├── tienkung2_lite/
│   ├── __init__.py                NEW — imports the hoisted constants,
│   │                               re-exports from .tienkung
│   └── tienkung.py                NEW — TALON_TIENKUNG_CFG
└── data/Robots/
    └── tienkung2_lite/            NEW — vendored USD/mesh, mirrors the
                                     unitree_a1/ vendoring pattern

talon_rl/tasks/manipulation/
└── tienkung_env/
    ├── __init__.py
    ├── config.py                  NEW — ObservationSpaceCfg/ActionSpaceCfg/
    │                               RewardVectorCfg for this task (reuses
    │                               talon_rl.config.PreferenceCfg as-is —
    │                               it's already dim-agnostic)
    ├── reward.py                  NEW — the 5 reward terms
    └── dummy_env.py                NEW — toy-physics smoke-test env
```

## Changes to existing code

`talon_rl/assets/unitree_a1/__init__.py` currently defines
`TALON_ASSETS_EXT_DIR`/`TALON_ASSETS_DATA_DIR`/`TALON_ASSETS_METADATA`/
`__version__` itself, even though the constants' *values* are already
robot-agnostic (`TALON_ASSETS_EXT_DIR = Path(__file__).resolve().parent.parent`
resolves to `talon_rl/assets/` regardless of which robot's `__init__.py`
computes it) — the module's own docstring already says the intent is "a
shared constant instead of each re-deriving it." Adding a second robot is
the trigger to actually make it shared: move those 4 definitions up to
`talon_rl/assets/__init__.py` (currently empty), and have both
`unitree_a1/__init__.py` and `tienkung2_lite/__init__.py` do
`from .. import TALON_ASSETS_DATA_DIR` (etc.) instead of redefining. No
behavioral change — same values, computed once instead of per-robot.

## `tienkung2_lite/tienkung.py`

Mirrors `unitree_a1/a1.py`'s shape exactly: take the reference project's
`TIENKUNG_CFG`-equivalent `ArticulationCfg`
(`TienKung-Lab-main/legged_lab/assets/tienkung2_lite/tienkung.py`), produce
`TALON_TIENKUNG_CFG = <reference>.replace(prim_path="{ENV_REGEX_NS}/Robot")`
with `spawn.usd_path` overridden to the vendored local copy
(`f"{TALON_ASSETS_DATA_DIR}/Robots/tienkung2_lite/tienkung2_lite.usd"`,
matching the source's own `usd/tienkung2_lite.usd`). Actuator gains
(`legs`/`feet`/`arms` `ImplicitActuatorCfg` stiffness/damping) are NOT
overridden this round — no equivalent of the A1's RMA-paper Kp/Kd retuning
exists yet for TienKung; keep the reference project's stock gains until a
reason to retune surfaces.

## `tienkung_env/config.py`

Same shape as `talon_rl/config.py`'s three dataclasses, own numbers:

```python
@dataclass(frozen=True)
class ObservationSpaceCfg:
    joint_pos_dim: int = 8   # 4 joint types x 2 arms
    joint_vel_dim: int = 8
    box_relative_pos_dim: int = 3   # box position relative to torso frame
    arm_contact_dim: int = 2        # binarized L/R arm-box contact
    prev_action_dim: int = 8
    preference_dim: int = 5         # one weight per reward-vector term

    @property
    def total_dim(self) -> int: ...  # sum of the above


@dataclass(frozen=True)
class ActionSpaceCfg:
    dim: int = 8   # target joint angle per arm DOF, PD-converted downstream


@dataclass(frozen=True)
class RewardVectorCfg:
    term_names: tuple[str, ...] = ("progress", "clearance", "energy", "impact", "smoothness")
    active: tuple[bool, ...] = (True, True, True, True, True)
    # reward-shaping constants: mirror talon_rl/config.py's pattern
    # (progress_std, impact_floor_eps-equivalent), rough starting points
    @property
    def dim(self) -> int: ...
```

No `command_dim` field (see Scope — no randomized target this round).
`PreferenceCfg` is NOT redefined here — import
`talon_rl.config.PreferenceCfg` directly, since its fields
(`dirichlet_alpha`, `max_delta_per_step`) and the functions in
`preference.py` that consume it are already fully dim-agnostic (operate on
`reward_cfg.dim`, `(N, dim)` batches) — confirmed by reading
`scripts/moppo/preference.py`.

Deliberately the *same* term names as the A1's `RewardVectorCfg`
(`progress`, `clearance`, `energy`, `impact`, `smoothness`) — same
multi-objective *shape*, retargeted semantics (below), underscoring that
this is the same methodology applied to a different embodiment/task, not a
different reward-vector design.

## Reward Vector (`tienkung_env/reward.py`)

| Term | A1 analog | TienKung semantics |
|---|---|---|
| `progress` | forward-velocity tracking | box height/position progressing toward the lift-and-hold target (chest height), exp-kernel like the A1's `progress_reward` |
| `clearance` | obstacle distance while walking | distance to obstacles while reaching for the box — same scripted-signal caveat as the A1's `clearance_reward` (placeholder until a real exteroception source exists) |
| `energy` | joint power (legs) | arm joint power (torque × velocity), same formula shape |
| `impact` | foot contact force | arm/box contact force, read from a `ContactSensorCfg` on the forearm links — a **privileged sim-only training signal** (reward computation runs at training time only, never needs sim-to-real, exactly like the A1's `foot_contact_force` precedent) |
| `smoothness` | action-rate/accel penalty | same formula, over the 8 arm DOF instead of 12 |

**Box mass/fragility is not part of the reward computation or the
observation this round** — inferring it online is the Adaptation Module's
job (Out of Scope). The `impact` term's sim-only `ContactSensorCfg` signal
gives the *reward* function privileged access to real contact force during
training (standard RL practice — training-time reward need not be
sim-to-real transferable); the *observation*'s `arm_contact_dim` field is
only a **binarized** contact flag (touching / not touching), which a real
robot could plausibly approximate even without an F/T sensor (e.g. from a
current-spike heuristic) — this keeps the observation space
deployment-plausible while keeping the reward computation simple for this
round.

**Box-dropped is a termination, not a reward term** — mirrors the A1's own
convention (catastrophic failure → `mdp/terminations.py`, not a per-step
reward penalty). Out of scope for the dummy env this round (no real
termination-condition plumbing exists yet — the dummy env's own reset
condition is just `horizon` elapsed, matching `DummyEnv`'s current
behavior); the real Isaac Lab env (a later increment) is where an actual
"box below height threshold" or "both-arms-lost-contact" termination term
gets added.

## `tienkung_env/dummy_env.py`

Mirrors `scripts/moppo/dummy_env.py`'s structure (`DummyEnv(gym.Env)` for
one lane + a `SyncVectorEnv`-backed `DummyTalonEnv` wrapper implementing
`BaseTalonEnv`) — reuse that *structure*, not that file's toy physics
(which is locomotion-specific: forward accel + landing impact). This
task's toy physics: `action` drives the 8 arm joint targets toward a
simple synthetic "box" state (e.g. a 1-D "how well-positioned + how much
force" toy model, analogous in spirit to `DummyEnv`'s
"action[0] drives forward accel, action[1] softens landing" but retargeted
to arms-closing-around-a-box) — exact toy kinematics are an implementation
detail for the plan, not fixed here; the requirement is only that it
produces every key `reward.py`'s 5 terms need (the `impact`/contact-force
key in particular, so the dummy env can exercise that reward term even
though there's no real contact physics — same spirit as the A1's own
`DummyEnv` fabricating a plausible `foot_contact_force` without real
ground contact).

## Testing

Same posture as the ConstraintManager and A1-terrain precedents established
this session: this machine has no Isaac Lab installed, so anything
importing `isaaclab.*` (the asset vendoring's `usd_path` construction is
plain Python/pathlib, no `isaaclab` import needed, so that part *is*
testable here) can't be executed. The dummy env and reward/config modules,
however, have **zero** `isaaclab` dependency (same as
`scripts/moppo/dummy_env.py` and `talon_rl/reward.py` today) — these
**can** and should be fully unit-tested here, mirroring
`tests/rewards/test_locomotion.py`/`tests/core/preferences/test_preferences.py`/`tests/core/envs/test_dummy_env.py`'s
existing patterns exactly (real assertions, run in this environment, no
Isaac Sim gate needed).
