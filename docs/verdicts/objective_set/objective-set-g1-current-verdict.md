# V3-G1 Current Verdict — Variable-Cardinality Training Substrate

Status: **FROZEN — G1 NOT YET INTERPRETABLE AS CARDINALITY GENERALIZATION**
Date: 2026-09-25

## Summary

V3-G0 passed exact representation and behavioral equivalence.

The first V3-G1 training implementation then failed, but the failure cannot be attributed to unseen-cardinality generalization because it already appears on training-seen cardinalities and the full m=4 anchor.

## Initial G1 training

Six preregistered runs were completed:

    G1-2 seeds 73101 / 73102 / 73103
    G1-3 seeds 73101 / 73102 / 73103

All six:
- completed 30 updates;
- were held-out-cardinality leakage free;
- kept PPO ratio error <= 6.1e-5;
- used the same objective-set architecture;
- preserved permutation invariance.

However the m=4 anchor failed in every run:

    T/A/O anchor PASS runs     0 / 6
    fresh critic PASS runs     0 / 6

Thus the initial G1 result is not a valid cardinality-generalization test.

## Authority result

Preference authority did not collapse.

At m=4, pairwise authority retention across six runs:

    0.965 .. 1.437

Tangent-Jacobian retention:

    0.830 .. 1.362

All are above the frozen 0.75 gate.

Therefore the failure is:

    authority present
    semantic correctness lost
    critic validity lost

not loss of preference influence.
## Temporal localization

Every run starts from the exact G0 model:

    u0:
        T/A/O semantic PASS
        critic EV mean = +0.0471
        critic negative fraction = 0.206

By u10, all six runs have already lost T/A/O aggregate validity and fresh critic validity.

Examples:

    G1-2 / 73101:
        u10 EV = -0.698
        T fail, A fail

    G1-2 / 73102:
        u10 EV = -2.437
        T/A/O all degraded

    G1-3 / 73101:
        u10 EV = -0.197
        semantic anchor already incomplete

    G1-3 / 73103:
        u10 EV = -1.390
        semantic anchor already incomplete

The failure is early, not a late-training durability failure.

## Critic-substrate confound

The first G1 trainer changed a validated Phase-1 requirement:

Phase 1:
    64-step representative support
    frozen critic body
    ridge head refresh, lambda=1

Initial G1:
    32-step on-policy GAE
    online critic SGD
    train critic body/value basis directly

This was not a clean carry-over of the Phase-1 critic compatibility contract.
## G1-C0 fixed-policy critic parity

A token-query version of the validated support/ridge critic procedure was tested with actor frozen.

Results:

    G1-2:
        actor drift              0
        fresh EV mean           +0.445
        negative EV fraction     0.0147
        mean abs bias            0.0464
        PASS

    G1-3:
        actor drift              0
        fresh EV mean           +0.433
        negative EV fraction     0.0179
        mean abs bias            0.0415
        PASS

Therefore the token-query critic representation has sufficient fixed-policy capacity across variable training-seen cardinalities.

## G1-C1 actor re-enable screen

Actor update was re-enabled for 10 updates while replacing online critic SGD with current-policy 64-step support/ridge refresh.

Authority and permutation remained valid.

But u10 failed:

    G1-2:
        T PASS
        A FAIL by survival 0.875
        O PASS
        S FAIL
        critic EV = -1.305
        negative EV = 0.800

    G1-3:
        T FAIL
        A FAIL
        O PASS
        S FAIL
        critic EV = -0.972
        negative EV = 0.744

Full 30-update C1 was therefore not authorized.
## G1-C2 post-hoc exhaustive critic refit

The C1 u10 actors were frozen and their critics were refit using exhaustive current-policy allowed-cardinality support.

Actor drift:

    exactly 0

Fresh critic improved materially:

    G1-2:
        EV -1.305 -> -0.127
        negative fraction 0.800 -> 0.388

    G1-3:
        EV -0.972 -> -0.129
        negative fraction 0.744 -> 0.519

But neither critic passed the frozen fresh-value gate.

Semantic endpoints were unchanged by construction.

Thus two facts now coexist:

1. critic support coverage / generalization remains insufficient once the actor moves away from G0;
2. actor semantic regression is already real and cannot be repaired by post-hoc critic fitting.

## Current causal interpretation

The current evidence supports:

    objective-set representation             VALID
    permutation invariance                   VALID
    variable-cardinality execution           VALID
    preference authority                     VALID
    fixed-policy token-query critic capacity VALID
    online G1 critic substrate               INVALID
    moving-policy critic generalization      INSUFFICIENT
    actor semantic retention under G1 update INVALID at u10

Therefore:

    held-out cardinality generalization      NOT YET TESTED CLEANLY
    G1 final claim                           NOT ESTABLISHED
    G2                                       BLOCKED
    G3                                       BLOCKED

The failed held-out m=2 results from the first G1 run must not be cited as evidence that cardinality generalization itself fails, because seen cardinalities and the m=4 anchor had already failed.
## Next allowed question

The next diagnostic should isolate actor credit/update geometry while holding value quality under tighter control.

A clean next gate is:

    fixed matched states + fixed current actor
    compare actor update direction produced by:
        A. token-query critic advantages
        B. direct finite-horizon Monte-Carlo active-objective returns

without applying either update first.

Question:

> Is the G1 actor semantic drift caused by inaccurate critic-derived credit, or does the active-set PPO objective itself push the policy away from T/A/O semantic directions even when return targets are critic-independent?

This should remain a read-only gradient/update-direction audit before another training branch is authorized.

No objective-specific patch, architecture change, reward redefinition, or held-out-cardinality evaluation is authorized yet.
