# Unitree A1 Locomotion Task

<!-- nav:start -->
[TALON package](../../../README.md)
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

## Extension rule

Add code here when it is specific to the A1 locomotion task or Isaac Lab scene/MDP definition.

If the code is reusable across tasks, prefer `talon_rl/` model/reward/curriculum modules or `scripts/rl/core/`.
