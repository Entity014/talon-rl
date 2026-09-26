# Authority-Isolated Survival / Robustness Localization Verdict

Status: **FROZEN — READ-ONLY AUDIT COMPLETE; ACTOR ROBUSTNESS FAILURE LOCALIZED**
Date: 2026-09-25

## Scope

No training was performed. The frozen AI-H1 actors at `u50` and `u75` were evaluated on the same semantic reset suites and preferences used by AI-C1.

The audit records real termination terms, time-to-failure, matched reset lanes, pre-step physical state, action saturation, pre-tanh magnitude, generated-family magnitude, trajectory-local family action effect, and local simplex-tangent action sensitivity.

## Survival result

`u50` survives every tested lane in suites 0–3 for T/A/O/S/C.

`u75` reproduces the AI-C1 failure exactly:

    suite 0: T/A/O/S/C = 1.000
    suite 1: T/A/O/S/C = 1.000
    suite 2: T = 1.000; A/O/S/C = 0.875
    suite 3: T/A/O/S/C = 0.875

All nine failures are `base_contact`, never timeout.
Failure timing is early and tightly clustered:

    suite 2: steps 15–19
    suite 3: steps 14–15

Suite 3 is especially diagnostic: the same reset lane (lane 0) fails for every preference. Suite 2 also shows lane-specific structure: O/S fail lane 2 while A/C fail lane 5.

This makes the failure broader than a preference-specific outlier and strongly implicates reset/state-regime sensitivity.

## Matched pre-failure window: u75 vs u50

Across the nine failed u75 trajectories, using the eight states immediately preceding first failure and the identical reset lane/time window at u50:

    metric                 u75 fail     u50 matched    ratio
    height                 0.1955       0.2010         0.973
    tilt (deg)             5.915        3.268          1.810
    |ang vel xy|           2.184        1.777          1.229
    action L2              3.424        3.211          1.066
    action-rate L2         0.326        1.375          0.237
    saturated coords       95.25%       77.43%         1.230
    pre-tanh L2            71.79        42.78          1.678
The action norm is close to the 12-D tanh ceiling `sqrt(12)=3.464`, while action rate collapses. This is consistent with a near-saturated action plateau rather than rapidly oscillating commands.

## Is authority magnitude itself the proximate cause?

The global generated-parameter norm is larger at u75:

    generated-family norm  2.261        1.513          1.495x

But trajectory-local preference authority does not spike before failure:

    family action effect   0.1525       0.2415         0.632x
    local tangent Jacobian 0.7881       1.3934         0.566x

Compared with surviving lanes from the same u75 rollout/time windows:

    family action effect   0.1525 fail  vs 0.2273 survive
    local tangent Jacobian 0.7881 fail  vs 1.2546 survive
    generated-family norm  2.261  fail  vs 2.261  survive

Early-window failure correlation is also weak for generated-family norm (r≈0.064) and negative for family effect / local Jacobian, while pre-tanh magnitude is more associated with later failure (r≈0.445).

Therefore the data do not support the simple causal claim `more local preference authority -> fall`.
## Dynamics / safety-margin signature

Relative to surviving u75 lanes in matched windows, failed lanes show:

    height                 0.900x
    tilt                   1.460x
    |ang vel xy|           1.332x
    action L2              1.010x
    saturated coords       1.033x
    pre-tanh L2            1.248x
    action-rate L2         0.334x

The strongest interpretable signature is therefore a reset-sensitive dynamics regime entered under a highly saturated actor, with reduced height margin and larger tilt/angular motion before base contact.

## Decision

Current blocker classification:

    authority allocation           PASS
    critic capacity                REPAIRABLE / PASS mechanistically
    local authority spike          NOT SUPPORTED as proximate failure cause
    preference-specific failure    SECONDARY / INCOMPLETE
    reset/state regime             STRONGLY SUPPORTED
    action safety margin           STRONGLY IMPLICATED
    semantic H2 / AI-C2            STILL BLOCKED

Do not reduce authority merely because the generated family is globally larger at u75.
Do not cap the preference pathway without a direct intervention test.
Do not resume semantic interpretation yet.

The next causal intervention, if this branch is continued, should preserve preference authority and test actor output/safety-margin robustness separately from authority ownership—for example a predeclared saturation-margin or state-regime robustness experiment with actor-authority invariance checks.
