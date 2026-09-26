# Authority-Isolated Critic/Support Compatibility Verdict

Status: **FROZEN — CRITIC BODY INSUFFICIENT UNDER LATE AUTHORITY-ISOLATED DISTRIBUTION**
Date: 2026-09-25

## Formal verdict
The read-only audit rejects a support-only explanation.

At AI-H1 u75, neither expanded nor dense current-policy head refitting restores the frozen Foundation V2 EV gate, and a held-out dense-support linear probe on the same frozen critic body also fails.

Therefore, under the predeclared decision rule:

> **CRITIC BODY INSUFFICIENT**

No actor parameter was changed.

## u75 head-refit results

Semantic evaluation suites, early / late EV:

    CURRENT_HEAD     +0.574 / -4.087
    SELECTED12       +0.644 / -2.012
    EXPANDED         +0.645 / -1.441
    DENSE_CURRENT    +0.522 / -7.632

Expanded support improves the saved head substantially, but late EV remains negative and late negative-EV fraction remains 0.35.
Dense current-policy support does not rescue the semantic distribution:

    DENSE_CURRENT late negative fraction = 0.413
    combined negative fraction           = 0.256

Thus simple support-size expansion is not sufficient.

## Frozen-body held-out probe
A fresh ridge head was fit on 75% of dense current-policy reset seeds and evaluated on the held-out 25%, with the critic body frozen.

At u75:

    early EV                 +0.620
    late EV                  -0.034
    late negative fraction    0.35
    combined negative frac    0.20

The failure is much milder than on semantic suites, but it still misses the frozen late-EV and negative-fraction requirements.

This is important: the critic body is not globally unusable, but its late-phase features are no longer sufficiently linearly predictive/generalizable under the stronger authority-isolated actor.
## Localization
The u75 late failure is concentrated rather than universal.

DENSE_CURRENT head on semantic suites:

    Tracking EV       +0.411
    Angular EV        -8.016
    Orientation EV    -0.002
    Smoothness EV    -22.921

EXPANDED head:

    Tracking EV       +0.491
    Angular EV        -2.317
    Orientation EV    +0.151
    Smoothness EV     -4.090

The dominant representation problem is therefore late Angular/Smoothness prediction, not all four value heads.

Held-out dense support shows the same direction at smaller magnitude:

    Tracking EV       +0.406
    Angular EV        -0.523
    Orientation EV    +0.244
    Smoothness EV     -0.263
## Reset-suite structure
For the u75 EXPANDED head, late semantic-suite EV is:

    suite 0   -5.737
    suite 1   +0.013
    suite 2   +0.095
    suite 3   -0.137

Suite 0 is the strongest failure, but the problem is no longer isolated to one reset suite because suites 1-3 also approach or cross zero.

At u50, by comparison, EXPANDED late EV was strongly negative mainly because of suite 0 while suites 1-3 remained clearly positive.

This supports a progression from localized distribution mismatch toward broader late-phase representation insufficiency by u75.

## Feature-space support shift
Using DENSE_CURRENT support:

    late nearest standardized distance
      u50   0.373
      u75   0.434

    late nearest-distance p95
      u50   0.628
      u75   0.735

    late PCA-whitened distance
      u50   1.395
      u75   1.798
The late evaluation distribution therefore moves farther from the dense support geometry by u75, even though the support is freshly generated from the same frozen policy checkpoint.

This indicates that stronger sole-path authority produces a broader / harder late value-function regime, not merely stale replay support.

## Relation to AI-H1
AI-H1 established:

    sole-path preference authority does not drift;
    authority instead grows strongly through u75.

The present audit establishes:

    the previously validated critic body cannot maintain late-phase value generalization in that stronger policy regime through head/support refitting alone.

Hence the two findings should remain separate:

    actor authority allocation        supported
    original critic representation    not compatible at late high-authority regime

## Decision
Do not reduce actor authority to recover critic EV.
Do not amend only support size or refresh cadence.
Do not run AI-H2 semantic evaluation yet.

Any continuation requires a new critic-capacity / critic-representation hypothesis while keeping the authority-isolated actor treatment fixed.

That is a methodological change and must be predeclared separately.
## Thesis wording
> A frozen-actor compatibility audit showed that the late critic failure under authority isolation could not be repaired by broader current-policy support alone. Re-fitting the original linear critic head on expanded or dense current-policy support left late explained variance negative, and a held-out dense-support probe with the critic body frozen also failed the predeclared late-phase criterion. The degradation was concentrated primarily in the Angular and Smoothness value heads and coincided with increased late feature-space distance. Thus authority isolation preserved and amplified preference control, but pushed the policy into a regime where the previously validated critic representation no longer provided adequate late-phase value generalization.
