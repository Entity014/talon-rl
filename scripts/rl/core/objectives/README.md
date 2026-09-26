# Objectives

<!-- nav:start -->
[Architecture](../../../../docs/methods/architecture/teacher-architecture.md) · [Train and run](../../README.md) · [Experiments](../../experiments/README.md) · [Research](../../../../docs/README.md) · [RL core](../README.md) · [Package](../../../../talon_rl/README.md)

[TALON RL](../../../../README.md) · [RL runner](../../README.md) · [RL core](../README.md) · [Objectives](README.md)
<!-- nav:end -->


## Purpose

Owns optimization semantics: policy loss, value loss, scalarization, clipping,
regularization, and future MORL objective strategies.

## Files

| File | Role | Status |
| --- | --- | --- |
| `core.py` | `TrainingObjective` Protocol plus `D3POObjective` | Stable extension point + tested adapter |

## Legacy Compatibility

The tested numerical implementation lives in `losses.py`. `D3POObjective` in `core.py`
delegates to `d3po_actor_loss`, so the OO layer does not redefine the formula.

## Adding an Objective

Implement `TrainingObjective.loss(batch, policy)`. Do not collect environment data or own
checkpointing here. Objective classes should be swappable without rewriting a trainer.
