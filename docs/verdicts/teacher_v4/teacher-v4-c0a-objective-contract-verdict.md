# Teacher V4 — V4-C0a Objective-Contract Port Verdict

Status: **PORT PASS; V4-C env = stock + e_t (C0a2 PASS); divisors not yet measured; V4-C num_envs pending**
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

## V4-C0a2 — V4-C environment: stock + e_t (decided 2026-09-26)

The Talon env turned out to differ from stock A1 flat in more than command
and terrain: physics substep (`sim.dt` 0.02, decimation 1 vs 0.005, 4),
episode length (4 s vs 20 s), terminations, events (push, wide DR), and the
action contract. Rather than editing the Talon env toward stock, V4-C gets
its own env built on the stock cfg:

- `Isaac-Talon-A1-V4C-v0`, cfg `TalonV4CEnvCfg(UnitreeA1FlatEnvCfg)` in
  `talon_rl/tasks/locomotion/a1_env/v4c_env_cfg.py`, plain `ManagerBasedRLEnv`.
- Additions only: the 12-D privileged `e_t` group (same order as the frozen
  contract) and the robot USD set to the repo's `a1.usd`, as every M0/V3
  script did.
- Stock DR only. Plant factors stock does not randomize read constant. The
  leg-length channel uses a separate `nominal_leg_length` term (1.0 by
  construction), so a variant env that loses `legScale` still raises.
- Action contract is stock (scale 0.25, Kp 25 / Kd 0.5), so C0c reduces to
  this verification.
- Talon placeholder DR ranges, push, rough terrain, Talon commands and leg
  variants are V4-D's treatment on `Isaac-Talon-A1-v0`, which keeps its
  T/A/O/S port.

Gate `scripts/rl/experiments/architectures/authority/teacher_v4/v4c_env_parity.py`,
report `runs/teacher_v4_c0a2_v4c_env_parity-2026-09-26/report.json`: **PASS.**

| check | result |
|---|---|
| full `to_dict()` diff vs fresh stock `UnitreeA1FlatEnvCfg` | only `observations.privileged`, `scene.robot.spawn.usd_path` |
| timing | `sim.dt` 0.005, decimation 4, step 0.02 s, 1000-step episodes |
| action contract (live) | scale 0.25, Kp 25, Kd 0.5 |
| policy obs / e_t | 48-D / 12-D, finite |
| varying e_t channels | payload mass only (stock `add_base_mass` −1…3 kg on trunk) |
| reward terms | the full stock set; V4 reads the T/A/O/S subset |

Constant e_t values: friction 0.8, Kp 25, Kd 0.5, leg length 1.0, joint
range 1.0, terrain height 0, dynamic friction 0.6, passive joint damping 0.

Consequence for the V4-C0 sampling contract: this env spawns one USD, so
`replicate_physics` can stay at the stock `True`. The reason for 2048 envs
(throughput under `replicate_physics=False`) does not apply to V4-C, and the
exact M0 contract (4096 × 24, 4 minibatches, 300 iterations) is feasible
again. The 2048 × 24 / 2-minibatch contract then belongs to V4-D. Pending
user decision.
