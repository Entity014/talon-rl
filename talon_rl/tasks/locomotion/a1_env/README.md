# Unitree A1 Locomotion Task

<!-- nav:start -->
[Architecture](../../../../docs/methods/architecture/teacher-architecture.md) · [Train and run](../../../../scripts/rl/README.md) · [Experiments](../../../../scripts/rl/experiments/README.md) · [Research](../../../../docs/README.md) · [RL core](../../../../scripts/rl/core/README.md) · [Package](../../../README.md)

[TALON RL](../../../../README.md) · [Talon Rl](../../../README.md) · [A1 Env](README.md)
<!-- nav:end -->

Isaac Lab task definition for TALON's Unitree A1 locomotion experiments.

## What lives here

| path | role |
|---|---|
| `a1_env.py` | environment class / task integration boundary |
| `a1_env_cfg.py` | scene, action, observation, reward, event, termination, and task configuration |
| `mdp/` | task-specific observation, event, curriculum, and termination terms |
| `terrain_config/` | terrain-generation configuration for locomotion experiments |

## Responsibility

This folder defines the **robot task**, not the training algorithm.

Training logic belongs under [`scripts/rl/core/`](../../../../scripts/rl/core/README.md), while reusable model architectures live under [`talon_rl/models/`](../../../models/README.md).

That separation lets the same task be exercised by different scalar, vector, preference-conditioned, and teacher/student training pipelines.

## Teacher-phase role

For the active [Teacher Architecture](../../../../docs/methods/architecture/teacher-architecture.md), this task is the source of:

- robot observations and previous-action context,
- privileged plant/environment factors during teacher training,
- objective/reward signals,
- actuator and environment transition dynamics.

The teacher architecture should consume these semantics without pushing trainer-specific abstractions back into the task package.

## Observation, action and randomization reference

Two A1 tasks are in use. Since 2026-09-26 they share the same 48-D policy observation. Action scaling, PD gains and the privileged group still differ:

| task id | defined in | used by |
|---|---|---|
| `Isaac-Velocity-Flat-Unitree-A1-v0` | Isaac Lab stock (`isaaclab_tasks/.../velocity/config/a1/flat_env_cfg.py`) | the canonical Phase-1 controller and its V3 objective-set successors (`artifacts/phase1_canonical_actor.pt`, [`deployment/phase1.py`](../../../deployment/phase1.py)), the D3 sim-to-sim transfer contracts, and the Phase-5 plant ensemble |
| `Isaac-Talon-A1-v0` | this folder ([`a1_env_cfg.py`](a1_env_cfg.py)) | `scripts/rl/train.py` MOPPO path with the RMA Env Factor Encoder |

Only `Isaac-Talon-A1-v0` has a privileged group `e_t`. Talon checkpoints trained before 2026-09-26 used a 51-D layout and do not load against the current one.

### Policy observation

**48-D, both tasks.** No `scale` terms and no corruption. The trainers normalize obs with a running `RunningMeanStd`. The reference layout is `build_canonical_obs` in `deployment/phase1.py`, `ObservationSpaceCfg` declares the same order, and `tests/core/runtime/test_sim2sim.py` checks that the MuJoCo adapter matches it.

| slice | term | dim |
|---|---|---|
| 0:3 | base linear velocity (root COM), body frame | 3 |
| 3:6 | base angular velocity, body frame | 3 |
| 6:9 | projected gravity, body frame | 3 |
| 9:12 | velocity command (v_x, v_y, omega_z) | 3 |
| 12:24 | joint position minus `CANONICAL_DEFAULT_Q` | 12 |
| 24:36 | joint velocity | 12 |
| 36:48 | previous action | 12 |

Joint order is `CANONICAL_JOINT_ORDER`, grouped by joint type: hips FL, FR, RL, RR, then thighs, then calves. The preference `w` is appended later by the trainer and is never part of the env observation.

Before 2026-09-26 the Talon layout was 51-D, following RMA's x_t. That layout had roll_pitch and binary foot contact but no base linear velocity. Two terms were dropped:

- `roll_pitch` duplicates projected gravity.
- Foot contact is not available on every target robot, and it can be inferred from q, q̇ and a_{t-1}. Rewards still read `foot_contact_force` from the transition dict.

Base linear velocity is privileged on hardware, so a Student policy needs an estimator for it.

### Action

Both tasks use 12-D joint-position targets, and PD control converts the targets to torque. The actuator is `DCMotorCfg`, which is explicit, so PhysX joint damping is passive only.

| | stock / canonical | Talon |
|---|---|---|
| policy output | `tanh` output in [-1, 1] | clip `ActorCritic.ACTION_CLIP = 3.0` |
| target | `default_q + 0.25 * a` | `default_q + 0.15 * a` |
| PD gains | Kp 25, Kd 0.5 (stock `UNITREE_A1_CFG`) | Kp 55, Kd 0.8 (RMA, `assets/unitree_a1/a1.py`) |
| effort / velocity limit | 33.5 N·m / 21 rad/s | same |

### Randomization

**Stock flat task, as installed.**

| what | mode | range |
|---|---|---|
| trunk mass | startup, add | -1 .. +3 kg |
| friction | startup | fixed: static 0.8, dynamic 0.6 |
| root pose / velocity | reset | stock `reset_base` ranges |
| velocity command | resample every 10 s | stock: v_x, v_y, omega_z in [-1, 1] |

CoM randomization, external force and push are all disabled for A1. The terrain is a flat plane.

**Phase-5 plant ensemble** ([`wrappers/plant_ensemble.py`](../../../wrappers/plant_ensemble.py)). This wrapper sits on top of the stock task. At every reset it draws a 3-D Sobol sample and overwrites the physics values set above. Draws that match a held-out plant from the manifest are skipped.

| axis | range | applied as |
|---|---|---|
| `mass_delta_kg` | -1.5 .. +3.0 | trunk mass = default + delta, inertia scaled by the same factor |
| `passive_blend` lp | 0 .. 1 | joint damping: hips lp, thighs and calves 2·lp; armature 0.01·lp on all joints |
| `contact_blend` lc | 0 .. 1 | static friction 0.8, dynamic friction 0.6 + 0.2·lc, restitution 0 |

**Talon task** (`EventCfg` in [`a1_env_cfg.py`](a1_env_cfg.py)). Most ranges are marked `[TBD]` placeholders in the code.

| what | mode | range | observed in `e_t` as |
|---|---|---|---|
| payload mass on trunk | reset, add | 0 .. 5 kg | `payload_extrinsics` (mass) |
| payload CoM | startup | x, y ±0.05 m; z ±0.02 m | `payload_extrinsics` (CoM) |
| friction | reset | static 0.4 .. 1.2; dynamic 0.4 .. 1.0 | `friction_extrinsic`, static friction only |
| actuator Kp, Kd | reset, scale | ×0.8 .. 1.2 each | `motor_power_extrinsic` |
| leg length | spawn-time USD variant | {0.85, 0.925, 1.0, 1.075, 1.15} | `leg_length_extrinsic` |
| joint range | reset, scale | ×0.8 .. 1.0 | `joint_range_extrinsic` |
| terrain | sub-terrain cell | `A1_ROUGH_TERRAINS_CFG` | `local_terrain_height`, which is the curriculum level and not the local height |
| (same friction event) | | | `dynamic_friction_extrinsic` |
| passive joint damping / armature | reset | blend lp 0 .. 1: hip damping lp, thigh/calf 2·lp, armature 0.01·lp (the plant-ensemble family) | `joint_damping_extrinsic` (mean PhysX damping) |
| push | interval 10–15 s | ±0.5 m/s xy | not observed (disturbance) |
| velocity command | reset | v_x -0.3 .. 1.0; v_y ±0.3; omega_z ±0.5 | not observed (command) |

With the default `payload_treatment` the privileged group `e_t` is 12-D. Under `noise_only` it is 8-D, because payload is removed from the group. See `ExtrinsicsCfg.dim`. The dynamic-friction and damping channels were added on 2026-09-26 so that the Phase-5 plant-ensemble axes are observable when the ensemble wraps this task.

## Extension rule

Add code here when it is specific to the A1 locomotion task or Isaac Lab scene/MDP definition.

If the code is reusable across tasks, prefer `talon_rl/` model/reward/curriculum modules or `scripts/rl/core/`.
