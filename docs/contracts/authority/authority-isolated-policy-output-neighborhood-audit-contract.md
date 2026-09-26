# Authority-Isolated Policy-Output Neighborhood Robustness Audit Contract

Status: **FROZEN BEFORE EXECUTION**
Date: 2026-09-25

## Question

Is the repaired controller sitting in an unusually narrow neighborhood of stable action sequences, such that small generic perturbations of its own policy output cause delayed closed-loop failure?

## Frozen actors

Compare three frozen policies:

    u50             robust reference
    u75             problematic late authority-isolated actor
    repaired-u10    projected + active tail-descent partial repair

No model parameters are updated.

## Evaluation support

Semantic suites:

    suite2 seed 840003
    suite3 seed 840004

Preferences:

    O / S / C

For each actor/preference/suite, vulnerability is measured only on lanes that survive the corresponding no-perturbation baseline. A secondary common-support analysis uses only lanes that survive baseline under all three actors.

## Perturbation domain

Primary perturbation is applied in pre-tanh actor-output space:

    z'_t = z_t + delta_t
    a'_t = tanh(z'_t)

To make budgets comparable across actors with different saturation, delta_t is rescaled at each step so that the realized RMS action-space deviation

    RMS(tanh(z_t + delta_t) - tanh(z_t))

matches the requested action-equivalent budget as closely as possible.

## Action-equivalent budget ladder

Same budgets for every actor, preference, suite, and lane:

    epsilon_a = 0.01
    epsilon_a = 0.02
    epsilon_a = 0.04
    epsilon_a = 0.08

These are deliberately in the range of the small action differences already shown to have causal effects in prior donor/residual audits.

## Temporal support

Perturbations are smooth over the full pre-contact control segment:

    t = 0..15

No Class-B-specific t6..10 window is hard-coded.

Temporal families include:
- constant
- linear ramp
- half-sine
- full-sine
- smooth two-lobe sequence

## Generic morphology-aware action bases

No lane, preference, or failure-joint identity is used.

Spatial bases are defined only from quadruped morphology:
- global/common mode
- left-right antisymmetry
- front-rear antisymmetry
- diagonal antisymmetry
- hip-vs-distal contrast
- joint-type hip mode
- joint-type thigh mode
- joint-type calf mode
- all-leg alternating morphology mode

Each basis is normalized before use.

## Isotropic/random controls

At every budget:
- fixed smooth random 12D perturbation sequences;
- same action-equivalent RMS budget;
- same temporal support.

## Primary gate

Evidence for a narrow endogenous control-law neighborhood requires:

1. baseline-valid support exists for all actors;
2. small generic perturbations produce delayed base-contact failures from baseline-surviving lanes;
3. vulnerability is materially greater for u75 than u50;
4. repaired-u10 reduces vulnerability relative to u75 on the same/common support;
5. morphology-aware/optimized perturbations are stronger than energy-matched isotropic random perturbations;
6. failures occur after perturbation has had time to alter the closed-loop trajectory, rather than only as direct clipping artifacts.

Desired ordering:

    vulnerability(u50) << vulnerability(repaired-u10) < vulnerability(u75)

A weaker but acceptable PASS is:

    u50 < repaired-u10 < u75

at one or more small budgets with recurrence across O/S/C.

## Stop rule

If small generic policy-output perturbations do not reveal a reproducible robustness ordering or do not cause delayed failures above random controls, stop residual-mechanism mining.

Do not:
- tune perturbations to a specific joint;
- tune to a specific preference;
- tune to a specific lane;
- increase budgets beyond the predeclared ladder;
- reopen Class-B feature mining.

A FAIL redirects the project to broader training/reset-distribution hardening and larger validation coverage rather than a new residual-specific robustness method.
