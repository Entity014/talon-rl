# `architectures/authority/teacher_v4`

<!-- nav:start -->
[Architecture](../../../../../../docs/methods/architecture/teacher-architecture.md) · [Train and run](../../../../README.md) · [Experiments](../../../README.md) · [Research](../../../../../../docs/README.md) · [RL core](../../../../core/README.md) · [Package](../../../../../../talon_rl/README.md)

[TALON RL](../../../../../../README.md) · [RL runner](../../../../README.md) · [Experiments](../../../README.md) · [Architectures](../../README.md) · [Authority](../README.md) · [Teacher V4](README.md)
<!-- nav:end -->

6 scripts. One line each, taken from the file's own docstring — edit the docstring, not this file.

| file | lines | tags | description |
|---|---:|---|---|
| `divisor_calibration.py` | 111 | class | T3-B objective-divisor protocol, re-run on a chosen env: abs-mean of each raw T/A/O/S objective under the M0 root policy. |
| `forward_sanity.py` | 173 | class | V4-B forward and invariance sanity for the untrained TeacherV4 on live Isaac-Talon-A1-v0 obs and e_t. |
| `leg_length_audit.py` | 142 | class | V4-B1 audit of the e_t leg_length channel: USD variants, per-env spawn, legScale lookup and physical leg geometry. |
| `objective_contract_audit.py` | 116 | class | V4-C0a objective-contract port check: the T/A/O/S reward terms on Isaac-Talon-A1-v0 match the stock A1 flat terms in config and kernel output. |
| `spawn_benchmark.py` | 124 | class | V4-B1-Fix1 benchmark of one Isaac-Talon-A1-v0 config: startup, VRAM, env throughput and leg-length variant spread under replicate_physics on/off. |
| `v4c_env_parity.py` | 116 | class | V4-C env gate: Isaac-Talon-A1-V4C-v0 equals stock Isaac-Velocity-Flat-Unitree-A1-v0 except the declared additions, and its e_t is 12-D. |

## Run directories these touch

- `runs/teacher_v4_b1_fix1_spawn_benchmark-2026-09-26`
- `runs/teacher_v4_b1_leg_length_audit-2026-09-26`
- `runs/teacher_v4_b_forward_sanity-2026-09-26`
- `runs/teacher_v4_c0a2_v4c_env_parity-2026-09-26`
- `runs/teacher_v4_c0a3_divisor_calibration-2026-09-27`
- `runs/teacher_v4_c0a_objective_contract-2026-09-26`
