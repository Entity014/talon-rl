# Rollout

<!-- nav:start -->
[RL runner](../../README.md) · [RL core](../README.md)
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
