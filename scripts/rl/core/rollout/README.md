# Rollout

<!-- nav:start -->
[Architecture](../../../../docs/methods/architecture/teacher-architecture.md) · [Train and run](../../README.md) · [Experiments](../../experiments/README.md) · [Research](../../../../docs/README.md) · [RL core](../README.md) · [Package](../../../../talon_rl/README.md)

[TALON RL](../../../../README.md) · [RL runner](../../README.md) · [RL core](../README.md) · [Rollout](README.md)
<!-- nav:end -->


## Purpose

Separates environment interaction and advantage estimation from optimization objectives.

## Files

| File | Role | Status |
| --- | --- | --- |
| `core.py` | Rollout/advantage Protocols plus `PerObjectiveGAE` | Stable extension point + tested adapter |

## Legacy Compatibility

The GAE formula lives in `gae_functional.py`. `PerObjectiveGAE` delegates to it and adds
explicit return-target construction. MOPPO rollout collection still lives
inside `MOPPOTrainer` pending a dedicated behavior-preserving extraction.

## Adding Rollout Logic

Collectors return batches; estimators convert rewards/values into advantages and targets.
Neither should decide optimizer steps.
