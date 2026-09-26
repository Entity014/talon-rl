# V3-G1 — Trajectory-Level Active-Set Semantic Credit Verdict

Status: **FROZEN — D: NO COMPACT GRADIENT-LEVEL EXPLANATION**
Date: 2026-09-25

## Question

Does the seen-support semantic failure admit a stable trajectory-level explanation based on:

- per-objective MC update direction;
- mixed active-set MC direction;
- mixed critic-derived direction?

The audit used only training-seen cardinalities and committed no parameter update.

Virtual directions were norm matched and evaluated with matched deterministic closed-loop rollouts at:

    H = 8, 16, 32, 64

for all seen active sets in both G1 folds.

## Primary Tracking result

### G1-2

Tracking semantic classification by horizon:

    H8
        g_T^MC             neutral
        g_mixed^MC         neutral
        g_mixed^critic     neutral

    H16
        g_T^MC             degrade
        g_mixed^MC         improve
        g_mixed^critic     improve

    H32
        all three          neutral

    H64
        g_T^MC             degrade
        g_mixed^MC         improve
        g_mixed^critic     improve

At H64:

    g_T^MC
        Delta normalized T = -0.002606
        Delta physical T   = -0.004796

    g_mixed^MC
        Delta normalized T = +0.003616
        Delta physical T   = +0.002287

    g_mixed^critic
        Delta normalized T = +0.004072
        Delta physical T   = +0.004220

This is the opposite of the hypothesis that mixed scalarization consistently sacrifices Tracking.
## Interpretation

The previous gradient audit established that critic-derived and MC-derived update directions differ materially.

This trajectory audit now shows that the semantic consequences of those directions do not form a stable cross-fold, cross-horizon mechanism.

In particular:
- mixed-objective sacrifice of T appears clearly in G1-3 at H32/H64;
- the same mechanism reverses in G1-2, where mixed directions improve T at H64 while T-only MC degrades it;
- per-objective sacrifice matrices also change sign across horizons.

Therefore none of the following is sufficient as a general explanation of the seen-support failure:

    critic error alone
    T-vs-mixed gradient opposition alone
    per-objective return optimization alone
    a single stable scalarization-sacrifice mechanism

The G1 training objective should instead be treated as semantically unstable on seen support under the current formulation.

## Stop rule consequence

Actor-gradient mechanism mining is CLOSED.

Do not:
- add another objective-specific gradient correction;
- tune scalarization using these traces;
- retrain G1 from a post-hoc repair;
- interpret failed held-out-cardinality results as cardinality-generalization failure;
- open G2.

The defensible current statement is:

> The objective-set representation itself is valid and preference authority remains present, but the current G1 learning formulation does not preserve semantic validity even on training-seen objective sets. Extensive critic, gradient, and trajectory-level audits did not identify a compact cross-fold mechanism sufficient to justify another targeted gradient-level intervention.

## Phase status

    V3-G0 representation equivalence            PASS
    G1 variable-cardinality execution           PASS structurally
    G1 preference authority                     PASS
    G1 fixed-policy critic capacity             PASS
    G1 seen-support semantic learning           FAIL
    compact critic/gradient mechanism           NOT ESTABLISHED
    cardinality generalization                  NOT CLEANLY TESTED
    G2 compositional generalization             BLOCKED
    G3 unseen objective identity                BLOCKED
### G1-3

Tracking semantic classification by horizon:

    H8
        g_T^MC             degrade
        g_mixed^MC         improve
        g_mixed^critic     improve

    H16
        g_T^MC             improve
        g_mixed^MC         improve
        g_mixed^critic     degrade

    H32
        g_T^MC             improve
        g_mixed^MC         degrade
        g_mixed^critic     neutral

    H64
        g_T^MC             neutral
        g_mixed^MC         degrade
        g_mixed^critic     degrade

At H32:

    g_T^MC
        Delta normalized T = +0.001901
        Delta physical T   = +0.006037

    g_mixed^MC
        Delta normalized T = -0.000678
        Delta physical T   = +0.000680

At H64:

    g_mixed^MC
        Delta normalized T = -0.001901
        Delta physical T   = -0.002242

Thus G1-3 contains a genuine T-vs-mixed sacrifice pattern at H32/H64, but that pattern does not transfer to G1-2.

## Horizon instability

The sign/classification is not stable even within one fold.

Examples:

G1-2:
    g_T^MC
        H16 degrade
        H32 neutral
        H64 degrade

G1-3:
    g_T^MC
        H8 degrade
        H16 improve
        H32 improve
        H64 neutral

    g_mixed^critic
        H8 improve
        H16 degrade
        H32 neutral
        H64 degrade

Therefore no single short/medium/long-horizon sign captures the mechanism.

## Survival confound at H64

For primary Tracking evaluations:

    G1-2 H64 min survival = 0.875
    G1-3 H64 min survival = 0.875

while H8/H16/H32 remain at 1.0.

Therefore H64 differences cannot be treated as clean semantic-only effects.

The decisive evidence for heterogeneity already exists before H64, especially at H16/H32, so the D verdict does not depend on the survival-confounded endpoint.
## Sacrifice-matrix result

The full per-objective sacrifice matrices are heterogeneous.

At H64, G1-2:

    T semantics under:
        g_T       degrade
        g_A       degrade
        g_O       improve
        g_S       degrade
        mixed MC  improve
        mixed cr  improve

    A semantics under:
        g_T       degrade
        g_A       degrade
        g_O       degrade
        g_S       degrade
        mixed MC  improve
        mixed cr  degrade

At H64, G1-3:

    T semantics under:
        g_T       neutral
        g_A       degrade
        g_O       degrade
        g_S       degrade
        mixed MC  degrade
        mixed cr  degrade

    A semantics under:
        g_T       degrade
        g_A       improve
        g_O       degrade
        g_S       improve
        mixed MC  degrade
        mixed cr  degrade

The matrix does not reveal one stable objective whose individual gradient consistently owns or resolves the seen-support semantic failure.

## Decision-tree outcome

Predeclared outcomes:

    A multi-objective interaction proximate cause     NOT ESTABLISHED
    B objective-return/semantic mismatch dominates   NOT ESTABLISHED
    C critic error + interaction stable mechanism    NOT ESTABLISHED

Observed:

    fold dependence          strong
    horizon dependence       strong
    sacrifice matrix         heterogeneous
    H64 robustness confound  present

Therefore:

    D_NO_COMPACT_GRADIENT_EXPLANATION

and:

    G1 retraining authorized     NO
    actor-gradient mining        STOP
    held-out cardinality claim   STILL NOT TESTED CLEANLY
    G2                           BLOCKED
## Scientific interpretation

The earlier gradient audit correctly showed that critic-derived and MC-derived update directions differ substantially.

This trajectory audit now shows that those local gradient differences do not map to one stable closed-loop semantic mechanism.

In particular:

- G1-3 supports a T-vs-mixed sacrifice interpretation at H32/H64;
- G1-2 shows the opposite relation at H16/H64;
- per-objective gradients can damage their own semantic objective;
- mixed gradients can sometimes improve an objective they locally oppose;
- effect signs change with rollout horizon.

Thus neither:

    local gradient cosine

nor:

    objective-specific MC return gradient

is sufficient to predict closed-loop semantic behavior under the current G1 formulation.

The strongest defensible conclusion is:

> The V3-G1 active-set learning objective is not semantically stable on training-seen support, and the instability does not admit a compact critic-, scalarization-, or single-objective-gradient explanation that transfers across folds and horizons.

This failure occurs before a clean zero-shot cardinality test can be interpreted.

## Stop rule

Per the preregistered contract, actor-gradient mechanism mining now stops.

Do not:
- create another gradient regularizer from these traces;
- patch Tracking specifically;
- tune scalarization from the sacrifice matrix;
- reopen G1 training;
- evaluate G2.

A future continuation would require a new externally motivated formulation-level hypothesis rather than additional mining of the same G1 data.

## Primary artifacts

- docs/contracts/objective_set/objective-set-g1-trajectory-semantic-credit-contract.md
- scripts/rl/objective_set_g1_trajectory_semantic_credit.py
- runs/objective_set_g1_trajectory_semantic_credit-2026-09-25/trajectory_semantic_credit.json
