# V3-G1 Variable-Cardinality Generalization Verdict

Status: **FROZEN — G1 FAIL; SYSTEMATIC CARDINALITY-2 SEMANTIC REGRESSION; G2 NOT AUTHORIZED**
Date: 2026-09-25

## Question

Can one unchanged objective-set-conditioned controller train on selected active-set cardinalities and generalize zero-shot to a cardinality never exposed anywhere in training?

Frozen folds:

    G1-2: train m={3,4}, test m=2
    G1-3: train m={2,4}, test m=3

Frozen seeds:

    73101, 73102, 73103

All six training runs completed under the frozen 30-update budget.

No held-out-cardinality leakage was recorded in any training report.

## Training substrate

Across all six runs:
- same objective-set architecture and token dimension;
- active-token-only reward scalarization;
- active-token-only critic queries/targets;
- active-set-local edge retention only;
- no objective-specific branch/head/loss;
- u30 checkpoint fixed in advance;
- PPO ratio invariant remained within 1e-4;
- permutation invariance remains structural.

Training-time termination was mostly zero, with isolated 0.125 events in some seeds. These did not determine the scientific verdict.
## G1-2 held-out m=2 screen

The predeclared stop rule allows G1 to stop before full continuum/G1-3 semantic evaluation when T/A/O semantic validity systematically degrades across variable cardinality.

### Seed 73101

Held-out m=2:
- T/A/O endpoint requirement: FAIL
- critic: PASS
- permutation: PASS
- survival: PASS

Examples:

    {T,A}: T FAIL, A FAIL
    {A,O}: A FAIL
    {A,S}: A FAIL
    {O,S}: O FAIL

Full-set m=4 anchor also regressed:

    T  objective=.25 physical=.50  FAIL
    A  objective=.00 physical=.00  FAIL
    O  objective=.75 physical=.75  PASS
    S  objective=.50 physical=.50  FAIL

m=4 authority remained nontrivial:

    mean pairwise action distance = 0.9320

m=4 critic remained valid.

### Seed 73102

Held-out m=2 again failed the T/A/O requirement.

Examples:

    {T,A}: T FAIL, A FAIL
    {T,O}: O FAIL
    {A,O}: A PASS, O PASS but critic FAIL
    {O,S}: O PASS but critic FAIL

The full-set m=4 anchor again regressed:

    T  objective=.75 physical=.50  FAIL
    A  objective=.75 physical=.75  PASS
    O  objective=.25 physical=.75  FAIL
    S  objective=.75 physical=.50  FAIL

m=4 authority remained nontrivial:

    mean pairwise action distance = 0.6257

m=4 critic failed the frozen validity threshold in this seed.
### Seed 73103 confirmation

The held-out pair {T,A} was rerun and frozen as an independent third-seed confirmation.

Results:

    T objective correctness      .25
    T physical correctness       .25
    T endpoint                   FAIL

    A objective correctness      .25
    A physical correctness       .25
    A endpoint                   FAIL

    center compromise            .375  FAIL

At the same time:

    critic                       PASS
    survival                     1.00
    permutation action drift     0
    permutation value drift      0
    mean pairwise authority      0.2526

Therefore the third seed confirms the same key pattern:

    preference response exists
    permutation invariance holds
    critic can remain valid
    survival can remain valid
    but held-out-cardinality T/A semantics are wrong

The same {T,A} held-out pair failed in all three independent training seeds.

## Stop-rule decision

The G1 contract states:

> if T/A/O semantic validity systematically degrades across variable cardinalities, stop before G2; do not patch objectives individually.

That condition is met.

Therefore:
- no full pairwise-continuum/interior sweep is required to rescue G1;
- G1-3 semantic evaluation is stopped;
- existing G1-3 training artifacts are retained but are not used to claim cardinality generalization;
- no objective/cardinality-specific patch is authorized;
- G2 is not authorized.
## Interpretation

G0 established that the objective-set representation itself is function-preserving.

G1 shows that this structural generalization is not sufficient for learned cardinality generalization.

The failure is not simply:

    "the network cannot accept m=2"

It can.

Nor is it simply:

    "preference authority disappeared"

It did not.

The stronger supported statement is:

> After generalized training on m={3,4}, the same permutation-invariant controller can execute unseen m=2 sets and retain measurable preference authority, yet the intended T/A/O semantic direction does not generalize reliably. In some seeds critic validity also degrades, while in other failed sets the critic and survival remain valid.

This separates three properties:

    set-interface validity
        != preference authority
        != semantic generalization

It also extends the Phase-1 finding:

    durable authority != semantic correctness

to the variable-cardinality setting.

## V3 status

    V3-G0 representation equivalence      PASS
    V3-G1 variable-cardinality            FAIL
    V3-G2 unseen combinations             NOT AUTHORIZED
    V3-G3 unseen objective identity       NOT AUTHORIZED / stretch

## Thesis wording

Recommended:

> A permutation-invariant objective-set interface could reproduce the fixed-index Phase-1 controller exactly, but this representation equivalence did not imply zero-shot cardinality generalization after learning. Across three independent G1-2 training seeds, the held-out two-objective set {Tracking, Angular Stability} failed the frozen semantic endpoint criteria despite preserved permutation invariance, survival, and measurable preference-conditioned action separation; critic validity remained acceptable in the third-seed confirmation. Thus variable-cardinality execution and preference authority were insufficient to guarantee semantic generalization across objective-set size.

## Primary artifacts

- docs/contracts/objective_set/objective-set-g1-variable-cardinality-contract.md
- scripts/rl/objective_set_g1_train.py
- scripts/rl/objective_set_g1_endpoint_screen.py
- scripts/rl/objective_set_g1_ta_seed3_confirm.py
- runs/objective_set_g1_2_seed73101-2026-09-25/endpoint_screen/endpoint_screen.json
- runs/objective_set_g1_2_seed73102-2026-09-25/endpoint_screen/endpoint_screen.json
- runs/objective_set_g1_2_seed73103-2026-09-25/endpoint_screen/ta_seed3_confirmation.json
- runs/objective_set_g1_verdict-2026-09-25/g1_verdict.json
