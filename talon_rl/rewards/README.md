# Rewards

<!-- nav:start -->
[Architecture](../../docs/methods/architecture/teacher-architecture.md) · [Train and run](../../scripts/rl/README.md) · [Experiments](../../scripts/rl/experiments/README.md) · [Research](../../docs/README.md) · [RL core](../../scripts/rl/core/README.md) · [Package](../README.md)

[TALON RL](../../README.md) · [Talon Rl](../README.md) · [Rewards](README.md)
<!-- nav:end -->


- `locomotion.py`: primary A1 reward-vector implementation.
- `baselines.py`: B0, M0, and V1B reconstruction/baseline helpers.
- `objectives.py`: normalized/raw objective-vector definitions used by later experiments.
- `directed_progress.py`: simulator-independent directed-progress state and update logic.

## Raw reward library (V4 teacher)

Every term is a stock IsaacLab reward term computed by the env's reward
manager, already multiplied by its weight. Weights are the stock A1 flat
values: `RewardsCfg` in IsaacLab `locomotion/velocity/velocity_env_cfg.py`,
overridden by `config/a1/rough_env_cfg.py` and `flat_env_cfg.py`. The only
added term is `action_jerk_l2`
([v4c_env_cfg.py](../tasks/locomotion/a1_env/v4c_env_cfg.py), env
`Isaac-Talon-A1-V4C-S1-v0` only). Roles and divisors live in
[objectives.py](objectives.py).

| term | weight | role | objective |
|---|---|---|---|
| `track_lin_vel_xy_exp` | 1.5 | objective | T |
| `track_ang_vel_z_exp` | 0.75 | objective | T |
| `ang_vel_xy_l2` | −0.05 | objective | A (= D, Dynamic Stability) |
| `flat_orientation_l2` | −2.5 | objective | O |
| `action_rate_l2` | −0.01 | objective in V4-C only | S (dropped in V4-C3) |
| `action_jerk_l2` | −0.01 | objective in V4-C2S-R1 only | S1 |
| `lin_vel_z_l2` | −2.0 | constraint (`CONSTRAINT_TERMS`) | none |
| `dof_torques_l2` | −2e-4 | regularizer (`REGULARIZER_TERMS`) | none |
| `dof_acc_l2` | −2.5e-7 | auxiliary (`AUXILIARY_TERMS`) | none |
| `feet_air_time` | 0.25 | auxiliary (`AUXILIARY_TERMS`) | none |
| `dof_pos_limits` | 0.0 | excluded | none |

`undesired_contacts` is disabled (None) for A1.

**How terms reach training.** The V4 trainer
(`scripts/rl/experiments/architectures/authority/teacher_v4/train_v4c.py`)
builds a [K, n_terms] matrix from `OBJECTIVE_TERMS`: objective k is the sum of
its weighted terms divided by that objective's divisor (T3-B abs-mean,
`NORMALIZATION_DIVISORS`). Terms with role constraint, regularizer, auxiliary
or excluded have zero columns. The env still computes them, but they are
**not** in any objective and **not** in the training reward.

Within an objective, the relative weight of its terms comes from the stock
weights (T = 1.5 · lin + 0.75 · yaw), and one divisor normalizes the sum. The
terms are not normalized one by one.

## Objectives

R_k = (Σ weighted terms of k) / divisor_k, higher is better. Divisors are
T3-B abs-means (`NORMALIZATION_DIVISORS`, `S1_DIVISOR`).

| label | name | R_k | divisor | used in |
|---|---|---|---|---|
| T | Command Tracking | (1.5 · lin_xy + 0.75 · yaw_z) / d | 1.71946 | V4-C, V4-C2S-R1, V4-C3 |
| A (= D) | Dynamic Stability | −0.05 · ang_vel_xy_l2 / d | 0.15591 | V4-C, V4-C2S-R1, V4-C3 |
| O | Upright Orientation | −2.5 · flat_orientation_l2 / d | 0.01563 | V4-C, V4-C2S-R1, V4-C3 |
| S | Control Smoothness | −0.01 · action_rate_l2 / d | 0.08311 | V4-C only |
| S1 | Smoothness (jerk) | −0.01 · action_jerk_l2 / d | 0.21491 | V4-C2S-R1 only |

V4-C3 trains on K = 3 (`--objectives TAO`). D is the objective-level name.
Its current realization is R_D = 1.0 · R_A + 0.0 · R_S; code and artifacts
keep the label A.

## Measurement library (not rewards)

[measurements.py](measurements.py) logs physical measurements that the
stock reward library does not cover. They exist for feature selection only:
no weight, no objective, not in any training reward. They cannot be
weight-0 reward terms, because IsaacLab's RewardManager does not compute a
term with weight 0. Values are unweighted, non-negative costs.

| measurement | tier | initial role | meaning |
|---|---|---|---|
| `base_lin_acc_z_l2` | 1 | objective / constraint candidate | (Δv_z world / dt)², vertical acceleration |
| `body_height_osc_l2` | 1 | constraint / V candidate | (z − EMA₀.₅ₛ(z))², oscillation about a moving mean, not posture |
| `foot_slip` | 1 | constraint candidate | Σ over feet in contact of ‖v_xy‖ |
| `foot_impact_l2` | 1 | constraint candidate | Σ over feet of ‖ΔF_contact‖² between policy steps |
| `dof_vel_l2` | 2 | regularizer candidate | Σ q̇² |
| `action_magnitude_l2` | 2 | regularizer candidate | Σ a² (effort, not rate) |
| `joint_power_abs` | 2 | regularizer candidate | Σ \|τ q̇\| |
| `root_z`, `v_z_world` | 0 | raw channels | targets (vertical excursion) |

Deferred as gait/style diagnostics, not objective candidates: foot
clearance, GRF balance / tracking, bound, foot gather.

How terms are grouped into objectives:
[objective selection](../../docs/methods/general/objective-selection.md).
