# AI-C2 Clean2 Interim Verdict

Status: **FAIL AS EXECUTED — CAUSAL INTERPRETATION NOT YET CAPACITY-SPECIFIC**
Date: 2026-09-25

## Clean2 endpoint

Both arms start from the robust control-u20 actor and actor optimizer state.

After 10 actor updates:

### Narrow critic arm

    pairwise authority retention      0.7723
    tangent Jacobian retention        0.6131
    authority gate                    FAIL

    frozen semantic min survival      1.00
    held-out min survival             1.00
    PPO ratio max error               0

    fresh H32 EV mean                 0.4471
    H32 negative fraction             0.000
    fresh MC64 EV mean                0.3424
    MC64 negative fraction            0.000

### Wide critic arm

    pairwise authority retention      0.7884
    tangent Jacobian retention        0.6627
    authority gate                    FAIL

    frozen semantic min survival      1.00
    held-out min survival             1.00
    PPO ratio max error               0

    fresh H32 EV mean                 0.4611
    H32 negative fraction             0.000
    fresh MC64 EV mean                0.3524
    MC64 negative fraction            0.0125

Both fresh critic audits contain isolated survival=0.875 cases, so the predeclared critic gate also fails on the fresh-survival subcriterion.

## Important causal issue

The paired run did not only change critic capacity.

Before actor learning, both critics were newly equilibrated on a common H32 support dataset.

For the wide arm this replaced the critic state that had already co-adapted with the robust control-u20 actor during the successful repaired continuation.

Therefore clean2 jointly changes:

    critic capacity
    +
    critic representation / credit geometry reset

relative to the known-compatible robust control-u20 system.

The result proves:

    newly re-equilibrated critics
    can drive authority loss during continued actor learning

but does not yet prove:

    wide critic capacity itself is incompatible

because the inherited control-u20 wide critic was already shown to remain compatible through the u10->u20 robust continuation.

## Next causal gate

Before another paired training run, perform a measurement-only initial credit-geometry audit on the same robust-u20 actor/rollouts:

Compare:

    A. inherited robust-u20 wide critic
    B. newly refit wide critic from clean2
    C. newly refit narrow critic from clean2

On exactly matched rollout states/actions/pre-tanh samples compute:

- objective-wise values
- vector GAE
- scalarized PPO advantages
- actor PPO gradient
- gradient cosine to inherited-wide reference
- gradient norm ratio
- preference/family parameter gradient share
- tangent-authority directional derivative / predicted authority change

If B already changes the actor credit gradient strongly relative to A, the clean2 authority collapse is a critic-reset compatibility problem, not a capacity result.

Only after this read-only audit should AI-C2 be rerun with a capacity-only initialization contract.
