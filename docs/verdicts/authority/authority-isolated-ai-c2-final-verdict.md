# AI-C2 Final Verdict

Status: **FAIL — CRITIC COMPATIBILITY PASSES, LONG-HORIZON ACTOR AUTHORITY PERSISTENCE FAILS**
Date: 2026-09-25

## Starting point

Authoritative robust checkpoint:

    runs/authority_isolated_coverage2_control-2026-09-25/model_20.pt

At this checkpoint:
- semantic-suite minimum survival = 1.00
- held-out minimum survival = 1.00
- fresh H32/MC64 critic gate passes
- preference authority is healthy

## Clean2 paired critic-capacity attempt

The first AI-C2 paired rerun re-equilibrated both narrow and wide critics on new H32 support before actor learning.

After 10 actor updates:

    narrow:
      pairwise retention     0.772
      tangent retention      0.613
      authority              FAIL

    wide:
      pairwise retention     0.788
      tangent retention      0.663
      authority              FAIL

Both arms retained:
- semantic survival = 1.00
- held-out survival = 1.00
- valid positive H32/MC64 EV
- exact PPO ratio contract

However, offline credit-surface audit showed that critic re-equilibration itself changed the actor credit geometry substantially relative to the inherited robust-u20 wide critic:

    refit-wide:
      value RMSE vs inherited       0.308
      value correlation             0.733
      preference-tangent cosine     0.458

    refit-narrow:
      value RMSE vs inherited       0.289
      value correlation             0.778
      preference-tangent cosine     0.488

Therefore clean2 was not capacity-only; it also reset critic/credit geometry.

## Capacity-only initialization audit

A narrow critic was distilled directly to the inherited robust-u20 wide critic.

Actor remained exact:

    max actor error = 0

Distillation fit loss reached:

    MSE ~ 8.5e-4

But the original-width critic could not meet the predeclared credit-equivalence gate:

    value correlation             0.977    target >= .98
    preference-tangent cosine     0.875    target >= .90
    tangent norm ratio            0.931

This provides evidence of a critic representation-capacity gap, but does not by itself establish the coupled actor-learning result.

No post-hoc distillation tuning was performed.

## Conservative wide-treatment confirmation

To remove the critic-reset confound entirely, the wide treatment was rerun using:

    exact inherited robust-u20 wide critic
    exact robust-u20 actor
    exact robust-u20 actor optimizer state
    global continuation schedule u21 -> u30
    same projected/tail repair
    same lambda=.95
    same reset distribution
    same anchor/reset schedule
    no critic-body refit

The critic body remained inherited and frozen; only the existing analytical head-refresh mechanism was used.

### Training endpoint

After 10 further actor updates:

    pairwise preference separation retention    0.8068
    simplex-tangent Jacobian retention          0.8512
    authority gate                              FAIL

Training rollouts:

    max termination fraction                    0
    PPO ratio max error                         0

Thus the actor remains dynamically robust during training, but preference authority decays over the longer continuation.

## Wide-treatment endpoint measurement

Frozen semantic suites:

    min survival        1.00
    failed lanes        0

Held-out reset suites:

    min survival        1.00
    failed lanes        0

Fresh reset-diverse audit:

    min survival                    1.00

    H32 EV mean                     0.3729
    H32 negative fraction           0.0625
    H32 mean absolute bias          0.0322

    MC64 EV mean                    0.2938
    MC64 negative fraction          0.0500
    MC64 mean absolute bias         0.1574

Therefore the wide critic remains valid under continued actor learning.

## Causal conclusion

AI-C2 no longer has a critic blocker.

The inherited wide critic:

    remains valid
    preserves robustness
    preserves PPO correctness

during u20 -> u30 continuation.

The failing gate is:

    long-horizon actor preference authority persistence

This is independent of:
- critic representation collapse;
- external robustness failure;
- reset-coverage failure;
- PPO ratio failure.

The result therefore refines the project state to:

    critic-capacity compatibility      PASS
    actor robustness compatibility     PASS
    long-horizon authority retention   FAIL
    full AI-C2 gate                    FAIL

## Narrow diagnostic arm

The narrow-distilled post-update control was not run after the wide treatment failed the primary authority gate.

This follows the predeclared conservative stop rule:

    if wide treatment itself fails authority,
    narrow contrast cannot authorize AI-C2.

The narrow capacity-init failure remains useful evidence that the original-width critic cannot fully reproduce the inherited wide credit surface.

## Decision

    AI-C2 full gate          FAIL
    AI-H2                    BLOCKED

Do not reopen:
- critic capacity mining;
- robustness mechanism mining;
- coverage expansion;
- adversarial disturbance training.

The next blocker is actor-side:

    preserve preference authority over continued learning
    after the robust u20 regime has already been reached.

Any next intervention must preserve:
- inherited wide critic
- robust control-u20 state
- projected/headroom repair
- PPO ratio contract
- survival contract

and change only the actor authority-persistence mechanism.

The next causal question is:

    Why does preference authority grow through u20
    but decay again during u20 -> u30
    despite valid critic and perfect robustness?
