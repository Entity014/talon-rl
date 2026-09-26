# Teacher V4 — V4-A / V4-B / V4-B1 Input-Contract Verdict

Status: **OPEN — V4-A PASS, V4-B PASS (to be rerun), V4-B1 bugs confirmed, plumbing fixed, spawn decision pending**
Date: 2026-09-26
Branch: `v4-a-teacher`

## Purpose

These stages check that the V4 privileged teacher is built correctly and
receives a correct input before any training. V4-C (training comparison
against V3) stays blocked until both the `e_t` contract and the action
contract are frozen.

## V4-A — structural model

Commit `4c9f401`, model `talon_rl/models/authority/teacher_v4.py`,
tests `tests/models/authority/test_teacher_v4.py`.

    h_t  = StateTrunk(x_t)             48 -> 256 (LayerNorm) -> 128 -> 16
    z_t  = EnvEncoder(e_t)             12 -> 256 -> 128 -> 8
    z_w  = rho(sum_i w_i phi(E(o_i)))  DeepSets objective set
    u_t  = policy([h_t, z_t]; theta(z_w))
           fixed 24 -> 256, family residual on 256 -> 128 -> 12
    c_t  = f_V(x_t, e_t, z_w),  V_i = MLP([c_t, q_i])

Decisions confirmed as the V4 default:

1. The critic has its own objective embedding and set encoder. A value loss
   never moves actor parameters. A shared embedding is a later ablation only.
2. The query head is `MLP([c_t, q_i])` with no raw `w_i`. Preference reaches
   the critic only through `z_w`.
3. `e_t` normalization lives outside the model, in the runner, with its
   statistics saved alongside the checkpoint.

Structural tests: shapes, log-prob consistency, permutation invariance,
zero-weight padding, cardinality 1–4, actor/critic gradient isolation in
both directions, and input validation. **PASS.**

## V4-B — forward sanity on live Isaac data

Commit `fa3e023`, script
`scripts/rl/experiments/architectures/authority/teacher_v4/forward_sanity.py`,
report `runs/teacher_v4_b_forward_sanity-2026-09-26/report.json`.

The untrained teacher ran on 2112 samples of `Isaac-Talon-A1-v0` (64 envs × 33 steps). `e_t` passed through `RunningNormalizer(12, center=True)`, the same normalizer MOPPO uses.

| check | result |
|---|---|
| obs 48-D, `e_t` 12-D | pass |
| `e_t` normalized (varying channels) | pass |
| `h_t`, `z_t`, `z_w`, `u`, action, value finite | pass |
| permutation, all 24 orderings | max diff 1.2e-7 |
| zero-weight padding | max diff 0.0 |
| plant authority (swap `e_t`) | median ‖Δz_t‖ 0.63, ‖Δa‖ 6.2e-3 |
| preference authority (swap `w`) | median ‖Δz_w‖ 0.14, ‖Δa‖ 7.2e-4 |
| set aliasing, 1024 mixtures | min ‖Δz_w‖/‖Δw‖ 0.11, p01 0.14 |
| `z_w` spectrum | 3 dominant singular values (2.9 / 1.9 / 0.92), matching the 3-DOF simplex |

**PASS**, with one limit: `e_t` channels 3 (`leg_length`) and 5
(`terrain_height`) were constant. Channel 5 is expected, because the
curriculum starts at level 0. Channel 3 triggered V4-B1. V4-B must be rerun
after the spawn decision, because channel 3 was wrong during this run.

At initialization, preference authority on the action is about 0.11× the
plant authority. This comes from the family-basis init scale, not from a
bug. V4-C should track whether this ratio grows during training.

## V4-B1 — `leg_length` channel audit

Commit `da50b74`, script
`scripts/rl/experiments/architectures/authority/teacher_v4/leg_length_audit.py`,
report `runs/teacher_v4_b1_leg_length_audit-2026-09-26/report.json`.

| link in the chain | finding |
|---|---|
| 5 USD variants | correct: `legScale` on `/a1`, link `xformOp:scale = s`, calf mass `0.166·s³`, PhysX thigh length `0.2·s` |
| per-env spawn | **bug**: all 64 envs reference `unitree_a1_leg_scale_1.075.usd` |
| `legScale` location | `/World/envs/env_i/Robot` |
| extrinsic lookup | **bug**: read `.../Robot/trunk`, fell back to 1.0 silently |
| reported vs simulated | `e_t[3]` reported 1.0 while PhysX simulated 1.075 |

Two separate bugs:

- **Bug A, plumbing:** `e_t` reported a false morphology.
- **Bug B, spawn:** the scene used `scene.replicate_physics = True`, which is
  the Isaac Lab default. That setting clones env_0's randomly chosen variant
  to every env. Isaac Lab warns about this at startup.

**Provenance consequence:** every `Isaac-Talon-A1-v0` run before this fix
simulated 1.075 legs in every env and had no leg-length domain
randomization. Phase-1 and V3 used the stock
`Isaac-Velocity-Flat-Unitree-A1-v0` with `a1.usd`, so they are not affected.

## V4-B1-Fix0 — plumbing fix

Commit `8c1dd98`, report `runs/teacher_v4_b1_leg_length_fix0-2026-09-26/report.json`.

`leg_length_extrinsic` now walks up from the articulation root to the first
prim that carries `legScale`. If no prim carries it, it raises instead of
returning 1.0. The audit is now a regression gate and exits 1 on failure.
Per env, it requires that the reported value matches the stage `legScale`,
the referenced variant file, and the PhysX thigh length `0.2·s`.

Rerun on 64 envs with the replicated scene: `e_t[3] = 1.075` everywhere,
matching physics. **PASS.**

## V4-B1-Fix1 — spawn benchmark

Commit `398bfc0`, script
`scripts/rl/experiments/architectures/authority/teacher_v4/spawn_benchmark.py`,
results `runs/teacher_v4_b1_fix1_spawn_benchmark-2026-09-26/` (12 JSON).

Each (`replicate_physics`, N) pair ran in its own process with zero actions and 200 timed steps. VRAM is the whole-GPU reading from nvidia-smi. The correctness gate requires that the PhysX thigh length equals `0.2·e_t[3]` in every env.

| N | startup s (r1 / r0) | peak VRAM MiB (r1 / r0) | env-samples/s (r1 / r0) | Δ | r0 variants |
|---:|---|---|---|---:|---|
| 64 | 2.7 / 2.9 | 3623 / 3615 | 8.6k / 8.5k | −1% | 5 |
| 256 | 2.8 / 3.6 | 3692 / 3683 | 26.4k / 26.1k | −1% | 5 |
| 512 | 3.3 / 4.6 | 3859 / 3820 | 39.4k / 38.0k | −4% | 5 |
| 1024 | 3.7 / 6.9 | 3981 / 3953 | 56.4k / 50.4k | −11% | 5 |
| 2048 | 5.4 / 11.9 | 4297 / 4290 | 73.9k / 56.6k | −23% | 5 |
| 4096 | 11.5 / 26.9 | 4901 / 5000 | 84.3k / 51.4k | −39% | 5 |

`r1` is `replicate_physics=True`; `r0` is `False`.

- `r0` gives all 5 variants at every N with a near-uniform spread (4096:
  823 / 831 / 810 / 845 / 787). `e_t[3]` matches physics in every env.
- The VRAM cost of `r0` is at most about 100 MiB. The stop rule (less than
  1 GiB free) never fired. The minimum free VRAM was 3.19 GiB.
- The cost of `r0` is throughput, and it grows with N. `r0` peaks at
  2048 envs. At 4096 it is slower than at 2048.

These numbers measure only the env step. They do not include policy forward
or update time, so the throughput gap will be smaller during training.

## Decision pending

Proposed canonical V4 env: **`replicate_physics=False`, `num_envs=2048`.**
This setting gives correct morphology diversity at the best `r0`
throughput. It runs at 67% of the throughput of the replicated 4096 setup.
The replicated 4096 setup is faster, but its morphology diversity is fake,
so it gives the policy a training exposure that does not match what the
method claims.

Open item tied to this decision: the PPO batch per iteration halves at 2048
envs. The two options are to double `num_steps_per_env` or to accept the
smaller batch. Freeze this together with the action contract.

## Next

1. Apply the spawn decision in `IsaacLabTalonEnvCfg`.
2. Rerun the V4-B1 audit with `require_diversity=True`.
3. Rerun V4-B with a varying `e_t[3]`.
4. Freeze the `e_t` contract and close this verdict.
5. Freeze the action contract (Talon 0.15 / clip 3.0 / Kp 55, Kd 0.8 versus
   canonical 0.25 / tanh ±1 / Kp 25, Kd 0.5), then V4-C.
