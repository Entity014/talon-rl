# `architectures/authority/teacher_v4`

<!-- nav:start -->
[Architecture](../../../../../../docs/methods/architecture/teacher-architecture.md) · [Train and run](../../../../README.md) · [Experiments](../../../README.md) · [Research](../../../../../../docs/README.md) · [RL core](../../../../core/README.md) · [Package](../../../../../../talon_rl/README.md)

[TALON RL](../../../../../../README.md) · [RL runner](../../../../README.md) · [Experiments](../../../README.md) · [Architectures](../../README.md) · [Authority](../README.md) · [Teacher V4](README.md)
<!-- nav:end -->

17 scripts. One line each, taken from the file's own docstring — edit the docstring, not this file.

| file | lines | tags | description |
|---|---:|---|---|
| `a_checkpoint_ladder.py` | 106 | class | Read-only A timeline on one V4-C run: at each saved checkpoint, the m=4 A endpoint (A-heavy vs center), A authority on the probes, survival, and the A critic's MC256 validity. |
| `a_credit_audit.py` | 137 | class | Read-only A-credit audit on V4-C checkpoints: per-objective advantage size, weighted surrogate share, and actor-gradient contribution in a training-like PPO batch. |
| `a_credit_quality.py` | 123 | class | Read-only A credit-quality check: does the GAE advantage that drives the actor agree with the realized long-horizon A advantage on the same states? |
| `anchor_evaluate.py` | 119 | class | V4-C seed-sensitivity m=4 anchor evaluation at iteration 300 (docs/contracts/teacher_v4/teacher-v4-c-seed-sensitivity-contract.md). |
| `ao_entanglement.py` | 131 | class | Read-only A-O entanglement diagnostic on a frozen V4-C checkpoint (m=4 set): reward overlap, behavioral separation, outcome separation, each ranked against the other objective pairs. |
| `critic_target_diagnostic.py` | 125 | class | Post-hoc V4-C C1-D0 diagnostic: critic EV against the registered truncated 32-step target versus the bootstrapped segment target, on the same G1 evaluation rollouts. |
| `divisor_calibration.py` | 111 | class | T3-B objective-divisor protocol, re-run on a chosen env: abs-mean of each raw T/A/O/S objective under the M0 root policy. |
| `forward_sanity.py` | 173 | class | V4-B forward and invariance sanity for the untrained TeacherV4 on live Isaac-Talon-A1-v0 obs and e_t. |
| `g1_evaluate.py` | 290 | class | V4-C G1 evaluation of one fold/seed at iteration 300: the V3 G1 protocol (objective_set_g1_evaluate.py) ported to TeacherV4 on Isaac-Talon-A1-V4C-v0. |
| `g1r_critic_evaluate.py` | 130 | class | V4-C G1-R: critic validity of the frozen V4-C checkpoints against a fixed 256-step Monte Carlo target, per docs/contracts/teacher_v4/teacher-v4-c-g1r-critic-contract.md. |
| `leg_length_audit.py` | 142 | class | V4-B1 audit of the e_t leg_length channel: USD variants, per-env spawn, legScale lookup and physical leg geometry. |
| `null_branch_feasibility.py` | 127 | class | V4-C2 null-only branch feasibility: how different are two branches restored from the same snapshot under the SAME preference, per horizon? |
| `objective_contract_audit.py` | 116 | class | V4-C0a objective-contract port check: the T/A/O/S reward terms on Isaac-Talon-A1-v0 match the stock A1 flat terms in config and kernel output. |
| `spawn_benchmark.py` | 124 | class | V4-B1-Fix1 benchmark of one Isaac-Talon-A1-v0 config: startup, VRAM, env throughput and leg-length variant spread under replicate_physics on/off. |
| `train_v4c.py` | 187 | trains class | V4-C trainer: TeacherV4 from scratch on Isaac-Talon-A1-V4C-v0 with objective-set PPO in the M0 shell. |
| `twins.py` | 77 | — | Same-state branching on Isaac-Talon-A1-V4C-v0: snapshot every env, then restore the snapshot into the same envs before each branch. |
| `v4c_env_parity.py` | 116 | class | V4-C env gate: Isaac-Talon-A1-V4C-v0 equals stock Isaac-Velocity-Flat-Unitree-A1-v0 except the declared additions, and its e_t is 12-D. |

## Run directories these touch

- `runs/objective_set_g0_structural_parity-2026-09-25`
- `runs/teacher_v4_b1_fix1_spawn_benchmark-2026-09-26`
- `runs/teacher_v4_b1_leg_length_audit-2026-09-26`
- `runs/teacher_v4_b_forward_sanity-2026-09-26`
- `runs/teacher_v4_c0a2_v4c_env_parity-2026-09-26`
- `runs/teacher_v4_c0a3_divisor_calibration-2026-09-27`
- `runs/teacher_v4_c0a_objective_contract-2026-09-26`
- `runs/update_functional_effect_audit-2026-09-24`
