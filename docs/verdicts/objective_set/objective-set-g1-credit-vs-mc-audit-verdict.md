# V3-G1 — Critic-Derived vs MC-Derived Actor Update Verdict

Status: **FROZEN — INCONCLUSIVE FOR CRITIC-ONLY CAUSE; ESCALATE TO TRAJECTORY-LEVEL ACTIVE-SET CREDIT**
Date: 2026-09-25

## Question

At the last semantically valid policy point (G0), does replacing token-query critic GAE with critic-independent MC64 credit remove the actor update direction that later destroys T/A/O semantics?

The audit was read-only:
- no update was committed;
- same on-policy H64 states/actions/pre-tanh samples;
- same old log-probabilities;
- same active sets and weights;
- same normalized objective rewards;
- same active-set PPO scalarization;
- only the credit source changed.

Audits were run independently on the training support of G1-2 and G1-3 using seed 73101.

## Mixed-gradient agreement

G1-2:

    cos(g_critic, g_MC)         0.4952
    ||g_critic||                3.455
    ||g_MC||                    6.304
    relative difference         0.870

G1-3:

    cos(g_critic, g_MC)         0.4228
    ||g_critic||                3.286
    ||g_MC||                    7.277
    relative difference         0.907

Thus critic-derived and MC-derived actor directions are materially different in both folds.

The critic is therefore not merely adding small noise to the same actor update.
## Per-objective credit agreement

Critic-vs-MC gradient cosine:

                G1-2       G1-3
    T           0.578       0.578
    A           0.760       0.772
    O           0.586       0.290
    S           0.641       0.632

The strongest disagreement is Orientation in G1-3.

However another pattern is also important: Tracking's own gradient opposes the mixed active-set gradient under both credit estimators.

Tracking-to-mixed cosine:

                critic      MC
    G1-2       -0.503     -0.400
    G1-3       -0.244     -0.287

Angular and Orientation generally align positively with the mixed gradient.

Therefore the audit exposes not only critic disagreement but also genuine multi-objective gradient interaction that remains present in the critic-independent arm.

## Functional effect

Norm-matched virtual parameter updates produce materially different policy-function perturbations.

At m=4 preferences, action-delta cosine between critic and MC virtual updates is approximately:

    G1-2:
        C  0.373
        T  0.529
        A  0.316
        O  0.241
        S  0.378

    G1-3:
        C  0.561
        T  0.508
        A  0.531
        O  0.433
        S  0.530

The parameter-gradient disagreement therefore survives into action space; it is not only a parameterization artifact.
## Semantic virtual-update result

### G1-2

MC64 direction:

    T   preserve / improve
    A   preserve / improve
    O   neutral within deadband
    degraded T/A/O count = 0

Critic-GAE direction:

    T   preserve / improve
    A   degrade
    O   degrade
    degraded T/A/O count = 2

This fold alone is consistent with critic-derived credit being a proximate blocker.

### G1-3

MC64 direction:

    T   degrade
    A   degrade
    O   neutral within deadband
    degraded T/A/O count = 2

Critic-GAE direction:

    T   preserve / improve
    A   degrade
    O   degrade
    degraded T/A/O count = 2

Thus the G1-2 semantic rescue does not transfer to G1-3.

Critic-independent MC credit can itself produce the same class of T/A/O semantic damage.

## Decision

The predeclared critic-blocker rule is NOT met because MC does not preserve/improve >=2 of T/A/O in both folds.

The active-set-PPO-geometry rule is also NOT met because critic-vs-MC mixed cosine is not >=0.8.

Therefore:

    critic-derived credit error              REAL
    critic-only explanation                  NOT SUFFICIENT
    active-set gradient interaction          PRESENT
    clean MC semantic rescue                 NOT TRANSFERABLE
    G1 retraining                            NOT AUTHORIZED
    held-out cardinality claim               STILL UNTESTED CLEANLY
    G2                                       BLOCKED

Frozen verdict:

    INCONCLUSIVE_ESCALATE_TRAJECTORY_CREDIT
## Causal interpretation

The evidence now rules out a simple story in which the G1 actor fails only because the token-query critic provides the wrong local update.

There are two simultaneous effects:

1. critic-derived and MC-derived gradients differ substantially;
2. even critic-independent MC optimization can move the policy in a semantically damaging direction under some active-set training support.

This is consistent with the Phase-1 distinction:

    objective-return improvement / optimization direction
        !=
    guaranteed semantic improvement

and with the earlier Phase-1 observation that finite objective-return improvements do not reliably imply semantic-margin improvements.

The G1 problem has therefore crossed from a pure critic-quality question into a trajectory-level active-set credit / multi-objective interaction question.

## Next authorized gate

Do not train again yet.

The next gate should be a trajectory-level active-set semantic credit audit that asks, for each active objective and mixed set:

    local / short-horizon return credit
        ->
    repeated closed-loop trajectory effect
        ->
    final semantic margin

The audit should use matched rollouts and compare:
- per-objective MC return advantage;
- mixed scalarized return advantage;
- trajectory-level semantic outcome;
- when objective-specific benefit is lost after mixing;
- whether Tracking's negative alignment with the mixed gradient predicts later semantic deterioration.

The purpose is to distinguish:

    objective-return / semantic mismatch
    vs
    multi-objective gradient interference
    vs
    trajectory-level closed-loop reversal

No objective-specific loss, scalarization change, architecture change, or new G1 training branch is authorized before that gate.

## Primary artifacts

- docs/contracts/objective_set/objective-set-g1-credit-vs-mc-audit-contract.md
- scripts/rl/objective_set_g1_credit_vs_mc_audit.py
- runs/objective_set_g1_credit_vs_mc_audit-2026-09-25/credit_vs_mc_audit.json
