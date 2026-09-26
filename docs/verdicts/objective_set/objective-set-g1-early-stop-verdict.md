# V3-G1 Variable-Cardinality Gate — Early Stop Verdict

Status: **FROZEN — G1 TRAINING SUBSTRATE FAILS ON SEEN FULL-SET ANCHOR; HELD-OUT CARDINALITY CLAIM NOT INTERPRETABLE; G2 BLOCKED**
Date: 2026-09-25

## What was run

Six preregistered G1 training runs completed:

G1-2:
- train cardinalities {3,4}
- held out m=2
- seeds 73101, 73102, 73103

G1-3:
- train cardinalities {2,4}
- held out m=3
- seeds 73101, 73102, 73103

All runs:
- completed 30/30 updates;
- passed leakage assertions at every update;
- used active-token critic queries only;
- used active-set-local edge retention only;
- preserved PPO ratio invariant (max <= 6.11e-5).

No held-out cardinality entered PPO, critic training, retention, replay/support, or checkpoint selection.

## Training substrate

Training-time maximum termination fractions:

    G1-2 seed 73101    0.000
    G1-2 seed 73102    0.000
    G1-2 seed 73103    0.125
    G1-3 seed 73101    0.000
    G1-3 seed 73102    0.125
    G1-3 seed 73103    0.000

These traces did not determine the verdict; final read-only anchor evaluation did.
## Seen full-set anchor screen

Because m=4 is present during training in both folds, it is the preregistered common seen-support anchor.

At u30:

    T/A/O semantic anchor PASS runs     0 / 6
    fresh critic PASS runs              0 / 6

Per-run T/A/O endpoint result:

    G1-2 / 73101    T FAIL  A FAIL  O PASS
    G1-2 / 73102    T FAIL  A FAIL  O FAIL
    G1-2 / 73103    T PASS  A PASS  O FAIL

    G1-3 / 73101    T FAIL  A PASS  O PASS
    G1-3 / 73102    T FAIL  A FAIL  O FAIL
    G1-3 / 73103    T FAIL  A FAIL  O FAIL

Fresh m=4 critic EV means:

    G1-2 / 73101    -0.225
    G1-2 / 73102    -0.442
    G1-2 / 73103    -1.328

    G1-3 / 73101    -0.268
    G1-3 / 73102    -0.840
    G1-3 / 73103    -1.873

All six violate the frozen critic gate.

## Authority did not collapse

On the frozen m=4 probe bank, all six runs retain the Phase-1 authority gate:

pairwise action-separation retention versus G0:

    1.437
    0.965
    1.242
    0.934
    1.237
    1.247

simplex-tangent Jacobian retention versus G0:

    1.362
    0.830
    1.339
    0.830
    1.131
    1.088

All are >= 0.75.

Therefore the seen-support failure is not preference-authority contraction.
## Full held-out evaluation status

A full G1-2 seed-73101 evaluation was completed before the broad anchor stop condition was confirmed.

Held-out m=2 sets:

    6 evaluated
    0 passed

However these held-out failures are **not interpreted as cardinality-generalization evidence**, because the same model also failed all seen m=3 sets and the seen m=4 anchor.

For that seed:
- authority gates passed on every evaluated set;
- permutation gates passed;
- endpoint survival was generally preserved;
- fresh critic EV was negative on every set;
- T/A/O semantics were inconsistent on both seen and held-out sets.

After the six-run m=4 anchor screen established systematic seen-support failure, the remaining expensive full held-out matrices were stopped under the preregistered G1 stop rule.

## Interpretation

The current G1 result does not answer:

> Can V3 zero-shot generalize to an unseen objective-set cardinality?

because the generalized training substrate first failed to preserve the known m=4 behavior that was explicitly present during training.

The causal state is:

    G0 representation equivalence            PASS
    permutation invariance                   PASS
    variable-cardinality execution           PASS
    held-out-cardinality leakage             NONE
    preference authority at m=4             PASS / retained
    fresh critic validity at m=4             FAIL 6/6
    T/A/O seen-anchor semantics              FAIL 6/6 runs
    cardinality generalization claim         NOT INTERPRETABLE
    G2                                      BLOCKED
## Critical implementation finding

The G1 trainer used direct online Adam updates of the shared token-query critic.

Phase 1's validated controller, however, required a stronger critic compatibility substrate based on representative/reset-diverse support and critic repair/refresh machinery.

Thus G1 generalized:
- objective representation;
- actor conditioning;
- value query interface;
- edge retention;

but did **not yet establish that the Phase-1 critic-compatibility requirement survives variable-cardinality training**.

Because actor PPO advantages depend on this critic, the critic collapse is a plausible upstream confound for the simultaneous semantic regression.

This does not prove that critic failure is the only cause.

## Next authorized gate

Before changing architecture or evaluating more held-out cardinalities:

    V3-G1-C0 — fixed-actor token-query critic isolation

Freeze the exact G0 actor and train only the token-query critic under the same fold-supported active-set distributions.

Question:

> Does the generalized critic remain valid on seen m=4 when actor/state visitation is held fixed with respect to policy parameters?

Decision:

- critic still fails -> generalized critic training/support substrate is independently insufficient;
- critic passes -> critic collapse in joint G1 is induced by actor/state-distribution drift or joint optimization, requiring a generalized Foundation-V2 critic-support/refresh path.

No objective-specific repair and no held-out-cardinality data are authorized.

## Primary artifacts

- docs/contracts/objective_set/objective-set-g1-variable-cardinality-contract.md
- scripts/rl/objective_set_g1_train.py
- scripts/rl/objective_set_g1_evaluate.py
- scripts/rl/objective_set_g1_anchor_screen.py
- runs/objective_set_g1_anchor_screen-2026-09-25/anchor_screen.json
- six G1 training run directories and provenance manifests
