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

How terms are grouped into objectives:
[objective selection](../../docs/methods/general/objective-selection.md).
