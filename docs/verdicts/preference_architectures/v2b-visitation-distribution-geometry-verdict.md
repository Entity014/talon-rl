# V2-B Visitation-Distribution Geometry Verdict

Status: FROZEN — VISITATION SHIFT DOES NOT EXPLAIN LATE ANGULAR COLLAPSE

Date: 2026-09-24

## Question

Does the u50->u75 on-policy state-distribution shift, by itself, rotate joint objective-gradient geometry enough to explain the late Angular semantic collapse?

## Causal design

Two source policies generated matched visitation states:
- source u50 policy
- source u75 policy

For each source state set, the evaluator policy was frozen and identical within the comparison. From each matched source state, the same evaluator policy, lambda=1.0, objective definition, continuation horizon, gradient computation, and common-random-number continuation were used.

The audit was replicated with two frozen evaluator policies:
- evaluator = u50
- evaluator = u75

Preferences: T/A/O/S. Phases: t8/t20/t36/t52. Seeds: 4.

Thus source visitation and evaluator policy were separated factorially.

## Visitation-only effect on Angular

When only source states change from u50 visitation to u75 visitation:

Evaluator u50, A-heavy delta in cos(combined,A):
- t8: +0.000
- t20: +0.005
- t36: -0.009
- t52: -0.020

Evaluator u75:
- t8: -0.003
- t20: -0.002
- t36: -0.006
- t52: +0.003

Angular weighted-share changes are also small (typically within about +/-0.03), and no consistent suppression pattern appears.

Therefore the u50->u75 visitation shift does not reproduce the Angular semantic collapse in frozen-policy gradient geometry.

## State-distribution shift is real

The source policies do reach measurably different states. Observation mean norms, joint-velocity norms, and previous-action norms change across phases and preferences. For example, early-phase joint-velocity mean-norm changes of roughly +0.48 to +0.61 occur for several heavy preferences.

Thus the null Angular result is not because u50 and u75 visitation are identical.

## Other objective interactions

Visitation effects are present in some regimes:
- late S-heavy states shift contribution toward Orientation; for evaluator u50 at t36, O share +0.112 and S share -0.082
- similar but smaller late S->O shifts appear with evaluator u75
- O-heavy combined-gradient magnitude can increase substantially on u75-source late states (up to ~1.65-1.88x depending on evaluator/phase)

These are real distribution interactions, but they do not explain the specific late Angular collapse.

## Complementary evaluator-policy effect at fixed visitation

Holding source states fixed while changing evaluator parameters from u50 to u75 also fails to show Angular credit loss.

For A-heavy states, late evaluator change often increases Angular share and/or reduces override:
- on u50-source t36: A share +0.091, cos(combined,A) +0.039
- on u75-source t36: A share +0.078, cos(combined,A) +0.043
- on u75-source t52: A share +0.032, cos(combined,A) +0.053

Thus the learned u75 policy still possesses locally healthy Angular optimization geometry on both u50-like and u75-like states.

## Causal status

- On-policy visitation shift exists: ESTABLISHED.
- Visitation changes gradient geometry in some objectives/regimes: ESTABLISHED.
- Visitation shift as primary explanation of Angular collapse: REJECTED.
- Static policy-parameter change as cause of local Angular credit loss: REJECTED.
- Local A gradient remains healthy at u75: RETAINED.
- Adam first-moment history: CONTRIBUTOR, not sufficient.

## Interpretation

The remaining failure sits downstream of static local gradient geometry.

> The policy can retain locally valid Angular gradients on both early-like and late-like state distributions while closed-loop Angular semantics still disappear after accumulated training.

This points to gradient-to-behavior realization on a nonlinear shared policy/dynamics manifold: finite parameter updates, cross-state coupling, and closed-loop trajectory consequences can change behavior even when the instantaneous local objective direction remains semantically correct.

## Decision

Do not reopen architecture, lambda, learning-rate, optimizer-reset, vector-lambda, or simple visitation-replay branches from this evidence.

The next decisive diagnostic, if continued, should test local-gradient-to-behavior realization directly: from the same frozen checkpoint/state distribution, apply a controlled small step along the measured Angular gradient (and matched control directions), then evaluate immediate and short-horizon Angular semantic change before allowing further on-policy updates. This would test whether a locally correct gradient actually produces the expected behavioral effect at finite parameter displacement.