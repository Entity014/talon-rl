# Teacher V4 — V4-C0a Objective-Contract Port Verdict

Status: **PORT PASS — kernels and weights match V3; divisors NOT frozen (wait on action contract); command/terrain distribution differences open**
Date: 2026-09-26
Branch: `v4-a-teacher`

## Decision

V4 keeps the V3 T/A/O/S objective semantics (formulas, signs, grouping,
weights) on the canonical V4 env. Normalization statistics are re-measured
on that env, not reused from the stock env.

## Port

`Isaac-Talon-A1-v0` had an empty `RewardsCfg` and no CommandManager.
`RewardsCfg` now carries exactly the five terms in
`talon_rl/rewards/objectives.py` `OBJECTIVE_TERMS`, with the stock
`Isaac-Velocity-Flat-Unitree-A1-v0` names, weights and std:

| term | objective | weight | kernel |
|---|---|---:|---|
| `track_lin_vel_xy_exp` (std 0.5) | T | 1.5 | ported: stock body, command from `v_command_buf` |
| `track_ang_vel_z_exp` (std 0.5) | T | 0.75 | ported: stock body, command from `v_command_buf` |
| `ang_vel_xy_l2` | A | −0.05 | stock function |
| `flat_orientation_l2` | O | −2.5 | stock function |
| `action_rate_l2` | S | −0.01 | stock function |

The ported kernels are in `talon_rl/tasks/locomotion/a1_env/mdp/rewards.py`.
The Phase-1 reward vector (`compute_reward_vector`) is unchanged; the
manager's scalar sum is unused.

## Checks

Script `scripts/rl/experiments/architectures/authority/teacher_v4/objective_contract_audit.py`,
report `runs/teacher_v4_c0a_objective_contract-2026-09-26/report.json`,
64 envs × 64 steps, uniform random actions.

| check | result |
|---|---|
| per term: Talon weight and params = stock `UnitreeA1FlatEnvCfg` (built, not hard-coded) | pass, all 5 |
| A/O/S use the identical stock function; T uses the port | pass |
| `OBJECTIVE_TERMS` grouping = exactly these 5 terms, manager has no others | pass |
| live kernel parity: manager term vs stock kernel on the same state via a CommandManager shim, 4088 non-reset lane-steps | max abs err ≤ 9.5e-7 (weight·dt/dt round-off) |
| V3 grouping builds a finite [N,4] objective vector from the Talon manager | pass |

Test suite: 314 passed.

## Why divisors are not frozen here

The T3-B divisors (`NORMALIZATION_DIVISORS`) are absolute means of each raw
objective under the **M0 root policy** (`runs/m0_1_seed0_2026-09-22/model_299.pt`,
stochastic actions clamped to ±1), 16 envs × 192 steps × 3 reset seeds
(`post_v2_t3b_scaling_audit.py` at tag `pre-reorg-2026-09-26`). On the
Talon env that policy acts through the Talon action contract (scale 0.15,
clip 3.0, Kp 55 / Kd 0.8), and `action_rate_l2` (S) is measured on the raw
policy action, so S scales directly with the action contract. The divisors
are therefore re-measured after V4-C0c freezes the action contract, with
the same T3-B protocol on the canonical env.

Preview only (uniform random actions, current Talon action contract, not
divisors): |T| 1.81, |A| 0.58, |O| 0.058, |S| 0.080. T3-B stock values:
1.72, 0.156, 0.0156, 0.083.

## Open: distributions that differ from the stock env

Kernel semantics now match, but two inputs to those kernels are distributed
differently on the canonical V4 env than on the stock env V3 used:

| | stock A1 flat (V3, M0) | Isaac-Talon-A1-v0 |
|---|---|---|
| command ranges | vx, vy, ωz ∈ [−1, 1], heading command | vx ∈ [−0.3, 1.0], vy ∈ [−0.3, 0.3], ωz ∈ [−0.5, 0.5] |
| command resampling | every 10 s, 2% standing envs | on reset only, 2 s stand phase |
| terrain | flat plane | rough generator with curriculum |

These shift what T and O measure in practice (tracking difficulty,
orientation on slopes). They are part of the env contract V4 is trained on,
and must be declared either as intended treatment or matched before V4-C is
compared numerically with V3.

## Also fixed

`IsaacAudit.execute` exited 0 even when a rollout raised, because
`app.close()` ends the process with status 0. Failed audits now exit 1
(probe: `SystemExit(1)` in `rollout` → process exit 1). Before this fix, the
"exits 1 on failure" gates in the V4-B1 audits could not signal failure
through the exit code; their recorded results were read from the report
JSON and are unaffected.
