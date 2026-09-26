# Phase-1 D3-C — Zero-Adaptation MORL Semantic Transfer Verdict

Status: **FROZEN — FAIL**
Date: 2026-09-26

## Question

After D3-A interface equivalence and D3-B dynamics viability pass, do the frozen T/A/O/S preference semantics transfer to MuJoCo without adaptation?

## Endpoint results

Tracking:

    objective correctness     0 / 4 = 0.00
    physical correctness      0 / 4 = 0.00
    survival                  1.00
    VERDICT                   FAIL

Angular stability:

    objective correctness     2 / 4 = 0.50
    physical correctness      2 / 4 = 0.50
    survival                  1.00
    VERDICT                   FAIL

Orientation stability:

    objective correctness     4 / 4 = 1.00
    physical correctness      4 / 4 = 1.00
    survival                  1.00
    VERDICT                   PASS

Smoothness:

    objective correctness     3 / 4 = 0.75
    physical correctness      3 / 4 = 0.75
    survival                  1.00
    CATEGORY                  S-valid

## Tracking transfer failure

Tracking failure is systematic rather than isolated.

T-heavy minus center:

    forward:
        objective delta   -0.000703
        physical delta    +0.205991

    turn-left:
        objective delta   -0.000673
        physical delta    +0.033526

    turn-right:
        objective delta   -0.000423
        physical delta    +0.019038

    lateral:
        objective delta   -0.000561
        physical delta    +0.103331

Higher objective delta should be positive and lower physical error should give a negative physical delta.

Both signs are reversed in all four suites.

Therefore the MuJoCo transfer produces a genuine semantic reversal for Tracking despite full rollout survival.

## Angular transfer

A-heavy is correct for:

    forward
    turn-right

and wrong for:

    turn-left
    lateral

Thus Angular semantic transfer is context-dependent rather than globally preserved.

## Orientation and Smoothness

Orientation transfers strongly across all four suites.

Smoothness, which is semantically inconsistent in the canonical deployment checkpoint, becomes valid under the MuJoCo transfer protocol.

This further supports the Phase-1/D1 conclusion that Smoothness semantic validity is strongly regime-sensitive.

## Preference-family structure

    center compromise                  0.8125   PASS
    continuum monotonicity             0.6641   PASS
    continuum endpoint-between         0.4965   FAIL
    minimum survival                   1.0000

Path-level continuum summary:

    T-A   monotonicity 0.7344   between 0.6667
    T-O   monotonicity 0.6875   between 0.5833
    T-S   monotonicity 0.6719   between 0.6250
    A-O   monotonicity 0.6562   between 0.4375
    A-S   monotonicity 0.5938   between 0.3333
    O-S   monotonicity 0.6406   between 0.3333

The preference family remains partially structured, but interpolation no longer reliably stays within endpoint behavior envelopes.

## Failure classification

This is not an interface failure:

    D3-A PASS

This is not a nominal locomotion-collapse failure:

    D3-B PASS
    all D3-C rollouts survive

The failure is therefore:

> **preference-semantic transfer failure under a physics-engine shift**

Specifically:

    T semantic transfer     FAIL
    A semantic transfer     FAIL / context-dependent
    O semantic transfer     PASS
    S semantic transfer     VALID in this regime
    center structure        PASS
    continuum structure     PARTIAL / endpoint-between FAIL

## Decision

    D3-C semantic transfer             FAIL
    D4 guarded hardware bring-up       BLOCKED

No D3 result authorizes policy, reward, actuator, friction, contact, or action-scale tuning inside D3.

Any adaptation study must be opened as a new explicitly named phase against this frozen zero-adaptation baseline.

## Source

    docs/contracts/transfer/phase1-d3c-semantic-transfer-contract.md
    scripts/rl/phase1_d3c_semantic_transfer.py
    runs/phase1_d3c_semantic_transfer/semantic_transfer_report.json
