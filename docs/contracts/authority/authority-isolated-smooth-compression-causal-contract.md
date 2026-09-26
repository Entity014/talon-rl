# Authority-Isolated Smooth Compression Causal Contract

Status: **FROZEN BEFORE EXECUTION**
Date: 2026-09-25

## Intervention

Frozen u75 actor only. No training.

Apply the extreme-selective pre-tanh transform:

    z' = c * tanh(z / c)
    a  = tanh(z')

This transform has unit first-order slope at z=0 and compresses only increasingly large logits.

Predeclared ladder:

    control: identity
    c = 3.0
    c = 2.5
    c = 2.0
    c = 1.75

The c=1.75 arm guarantees transformed logits are bounded below the |a|=0.95 saturation threshold because tanh(1.75) < 0.95.

## Evaluation

Same frozen semantic support as the previous saturation audit:
- reset seeds 840001..840004
- T/A/O/S/C
- 8 environments
- 64 steps

Fixed-probe authority:
- pairwise preference action separation
- simplex-tangent action Jacobian

Behavior deviation:
- mean/p95 action L2 deviation from unmodified u75

## Frozen causal gate

A compression arm supports an authority-preserving saturation mechanism only if:
1. minimum survival >= 0.95;
2. saturation fraction is lower than control;
3. pairwise authority retention >= 0.90;
4. tangent Jacobian retention >= 0.90.

The weakest compression (largest c) satisfying all criteria is primary.

If survival improves but authority falls below 0.90, the intervention is not accepted as authority-preserving.
If large headroom is created but survival remains blocked, saturation alone is insufficient and the remaining state-regime failure requires a different robustness mechanism.
