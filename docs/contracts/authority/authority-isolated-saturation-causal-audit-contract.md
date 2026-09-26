# Authority-Isolated Saturation Causal Audit Contract

Status: **FROZEN BEFORE EXECUTION**
Date: 2026-09-25

## Question

Does the frozen AI-H1 u75 actor fail on semantic reset suites because its output operates too deeply in the tanh-saturated regime, rather than because preference authority itself is excessive?

## Intervention

No learning, optimizer step, critic use, reward change, reset change, or parameter update is permitted.

For the frozen u75 pre-tanh mean z(s,w), evaluate:

    z' = alpha z
    a  = tanh(z')

with the predeclared ladder:

    alpha = 1.00, 0.90, 0.80, 0.70

alpha=1.00 is the exact frozen-policy control.

## Evaluation support

Use the same four semantic reset suites as AI-C1:

    seeds = 840001, 840002, 840003, 840004
    preferences = T/A/O/S/C
    num_envs = 8
    horizon = 64

## Measurements

Primary robustness:
- survival
- real termination reason
- time to first failure

Mechanism:
- pre-tanh L2 norm
- action saturation fraction, |a_i| >= 0.95
- action L2
- action-rate L2
- physical height, tilt, and xy angular velocity

Authority preservation on the fixed probe corpus:
- mean pairwise preference action separation
- simplex-tangent action Jacobian Frobenius norm

Behavior preservation:
- mean and p95 action L2 deviation from alpha=1.00 on identical fixed probe states/preferences

## Frozen decision rule

A treatment alpha supports the saturation-causal hypothesis only if all are true:

1. semantic-suite minimum survival >= 0.95;
2. saturation fraction is lower than alpha=1.00;
3. fixed-probe pairwise preference separation retention >= 0.90;
4. fixed-probe simplex-tangent Jacobian retention >= 0.90.

Action deviation is reported explicitly as a boundedness / interpretation metric but is not used to retroactively choose a threshold.

The smallest intervention (largest alpha below 1.00) satisfying all criteria is the primary causal candidate.

If survival does not recover despite materially lower saturation, saturation is treated as a correlate rather than an established proximate cause.

If survival recovers only after authority retention drops below 0.90, the audit does not support an authority-preserving saturation fix.

## Prohibitions

- no actor training
- no critic fitting
- no authority cap
- no preference-path modification
- no generic action penalty
- no semantic H2 claim
- no threshold relaxation after observing results
