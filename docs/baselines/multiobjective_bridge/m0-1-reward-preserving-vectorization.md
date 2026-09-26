# M0.1 — reward-preserving vectorization on the L0-A substrate

Status: **DESIGN DRAFT — training not authorized**

## Question

Can the stable stock Isaac Lab A1 flat formulation be represented as a reward
vector without changing the locomotion problem? M0.1 is a bridge experiment,
not preference conditioning. It must reproduce the stock scalar reward before
any MOPPO state or preference input is introduced.

## Frozen substrate inheritance

- Task: `Isaac-Velocity-Flat-Unitree-A1-v0`.
- Stock A1 flat observations, action semantics, reset/events, command manager,
  termination, physics rate, and `rsl_rl` PPO remain unchanged.
- No Gate-0B reward, B0/B1 trainer, RMA, terrain variation, exteroception, or
  preference state is allowed.
- Reference weights are fixed to one; no coefficient sweep is permitted.

## Objective vector

The vector contains the already-weighted stock reward terms, grouped only for
interpretability:

| component | stock terms included |
|---|---|
| `progress` | `track_lin_vel_xy_exp`, `track_ang_vel_z_exp` |
| `efficiency` | `lin_vel_z_l2`, `ang_vel_xy_l2`, `dof_torques_l2`, `dof_acc_l2`, `action_rate_l2` |
| `contact` | `feet_air_time`, `undesired_contacts` (zero when disabled by flat config) |
| `balance` | `flat_orientation_l2` |
| `limits` | `dof_pos_limits` |

For every transition, the required invariant is:

```text
scalar_stock_reward == sum(reference_weights * reward_vector)
reference_weights = [1, 1, 1, 1, 1]
```

Equality must be checked before reduction, per environment and per step, within
declared floating-point tolerance. Term values must come from the stock reward
manager's weighted term outputs, not reimplemented approximations.

## Acceptance gates

1. Unit test: grouping and reconstruction are exact on finite synthetic term
   tables, including disabled zero terms.
2. Isaac smoke: real stock A1 flat rollout emits vector and scalar rewards,
   with max reconstruction error below `1e-6`, finite values, and unchanged
   observations/actions/termination.
3. Short training smoke: vectorized path uses the same stock PPO config and
   reaches the same lifecycle/checkpoint schema; no preference/MOPPO state.
4. Only after these pass may a fresh 3-seed M0.1 confirmation be authorized.

M0.1 passes only if locomotion sanity remains consistent with L0-A and the
      reconstruction invariant holds across all seeds. M0.2 preference
conditioning remains closed until then.
