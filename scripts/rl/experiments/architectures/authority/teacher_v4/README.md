# `architectures/authority/teacher_v4`

<!-- nav:start -->
[Architecture](../../../../../../docs/methods/architecture/teacher-architecture.md) · [Train and run](../../../../README.md) · [Experiments](../../../README.md) · [Research](../../../../../../docs/README.md) · [RL core](../../../../core/README.md) · [Package](../../../../../../talon_rl/README.md)

[TALON RL](../../../../../../README.md) · [RL runner](../../../../README.md) · [Experiments](../../../README.md) · [Architectures](../../README.md) · [Authority](../README.md) · [Teacher V4](README.md)
<!-- nav:end -->

3 scripts. One line each, taken from the file's own docstring — edit the docstring, not this file.

| file | lines | tags | description |
|---|---:|---|---|
| `forward_sanity.py` | 173 | class | V4-B forward and invariance sanity for the untrained TeacherV4 on live Isaac-Talon-A1-v0 obs and e_t. |
| `leg_length_audit.py` | 142 | class | V4-B1 audit of the e_t leg_length channel: USD variants, per-env spawn, legScale lookup and physical leg geometry. |
| `spawn_benchmark.py` | 124 | class | V4-B1-Fix1 benchmark of one Isaac-Talon-A1-v0 config: startup, VRAM, env throughput and leg-length variant spread under replicate_physics on/off. |

## Run directories these touch

- `runs/teacher_v4_b1_fix1_spawn_benchmark-2026-09-26`
- `runs/teacher_v4_b1_leg_length_audit-2026-09-26`
- `runs/teacher_v4_b_forward_sanity-2026-09-26`
