# Authority-Isolated Smooth Compression Causal Verdict

Status: **FROZEN — OUTPUT SATURATION IS A STRONG CAUSAL CONTRIBUTOR, BUT NOT SUFFICIENT ALONE**
Date: 2026-09-25

## Intervention

Frozen u75 actor only. No training.

    z' = c * tanh(z/c)
    a  = tanh(z')

with control, c=3.0, 2.5, 2.0, 1.75.

This transform preserves unit first-order slope around zero while selectively compressing extreme logits.

## Robustness result

    arm       min survival   total failed lanes
    control      0.875              9
    c=3.0        0.875              6
    c=2.5        0.875              3
    c=2.0        0.875              1
    c=1.75       0.875              3

At c=2.0, all previous failures recover except:

    suite 3 / C / lane 0
    base_contact at step 15

At c=1.75, suite 3 O/S/C lane 0 fail at step 15.

Thus the response is strong but non-monotonic at the strongest compression.

## Saturation and authority

    arm       sat frac   pairwise retention   Jacobian retention
    control     0.9191          1.000               1.000
    c=3.0       0.9037          0.983               0.981
    c=2.5       0.8939          0.973               0.971
    c=2.0       0.8588          0.953               0.952
    c=1.75      0.0000          0.936               0.935

All smooth-compression arms preserve both fixed-probe authority measures above 90%.

At c=2.0:
- pairwise authority retention = 95.3%
- tangent Jacobian retention = 95.2%
- fixed-probe mean action L2 deviation = 0.131
- p95 deviation = 0.147
- failed lanes reduce from 9 to 1

At c=1.75:
- measured saturation fraction is exactly zero
- authority is still >93%
- failures increase from 1 to 3

## Interpretation

The predeclared primary gate does not formally pass because every arm still has minimum survival 0.875.

However, the mechanism is no longer merely correlational.

Selective extreme-logit compression causes a large robustness improvement while preserving learned preference authority:

    9 failures -> 1 failure
    authority retention ~95%

This is strong evidence that the late u75 output operating regime / insufficient action headroom is a causal contributor to the robustness blocker.

It is not the complete explanation. Eliminating the |a|>=0.95 saturation metric entirely at c=1.75 does not eliminate failure and actually worsens the best c=2.0 result. Therefore:
- saturation fraction itself is not a sufficient scalar target;
- generic stronger compression is not monotonically better;
- some state/preference regime still needs adequate action magnitude/geometry as well as headroom.

The residual c=2.0 failure is highly localized to suite 3, center preference, lane 0.

## Decision

Current causal picture:

    preference authority ownership      PASS
    critic representation               REPAIRABLE
    output/headroom robustness          CAUSAL CONTRIBUTOR — STRONGLY SUPPORTED
    pure saturation-only explanation    REJECTED AS SUFFICIENT
    generic action shrinking             NOT SUPPORTED
    semantic H2 / AI-C2                  STILL BLOCKED

Do not use a generic ||a|| penalty.
Do not reduce preference authority.
Do not target zero saturation as an objective.

If continuing the branch, the training-side repair should preserve the learned policy-family authority and discourage pathological extreme logits / loss of local headroom without forcing globally small actions. It must be validated against the residual suite-3 center-state regime and re-check fixed-probe authority before AI-C2.
