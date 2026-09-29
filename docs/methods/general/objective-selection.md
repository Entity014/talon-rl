# Objective Selection by Relevance–Redundancy Analysis

<!-- nav:start -->
[Architecture](../architecture/teacher-architecture.md) · [Train and run](../../../scripts/rl/README.md) · [Experiments](../../../scripts/rl/experiments/README.md) · [Research](../../README.md) · [RL core](../../../scripts/rl/core/README.md) · [Package](../../../talon_rl/README.md)

[TALON RL](../../../README.md) · [Documentation](../../README.md) · [Methods](../README.md) · [General](README.md)
<!-- nav:end -->

How the MORL objective set is chosen. The method adapts the feature-selection
principle of maximum relevance and minimum redundancy (mRMR) from features to
objectives. An objective is kept when it is relevant (it has a controllable
behavioral meaning) and not redundant with the objectives already kept
(it adds a controllable behavior dimension they do not). Objectives are
**not** kept or dropped by how well a policy scores on them.

## Objectives are behavioral axes, not reward terms

The measured raw terms, their weights and current roles are listed in the
[raw reward library](../../../talon_rl/rewards/README.md#raw-reward-library-v4-teacher).

    USER LEVEL        w_i            which behavior to trade toward (preference, per episode)
    OBJECTIVE LEVEL   R_i = Σ_k α_ik r̃_ik,  Σ_k α_ik = 1     fixed internal composition
    RAW LEVEL         r̃_ik           individual measured signals, each normalized (divisor)

An objective is one controllable behavioral axis. Several normalized raw
terms may define it; tracking is already one objective from two terms, for
example. The weights α are a fixed part of the formulation, set before
training and never tuned on results. The user preference w acts only at the
objective level. Selection (below) decides which axes deserve their own w.
If a raw signal turns out redundant with an axis, it becomes a candidate
constituent of that axis. That is the case for smoothness under Dynamic
Stability, not a discarded concept.

## Relevance and redundancy, operationalized

| feature-selection idea | objective analogue | measure (per pair i, j) |
|---|---|---|
| relevance | raising w_i moves objective i in its intended direction | self-direction S_i(i⁺) − S_i(reference) |
| correlation / redundancy | the objectives' outcomes co-vary | Spearman ρ of S_i, S_j per step and per 32-step window |
| unique support | states exist that are good on one and bad on the other | median-split off-diagonal quadrant mass (0.5 = independent) |
| dimensionality | the feasible outcome cloud is 2-D rather than a line | PC1 variance share of standardized outcomes (0.5 = 2-D, 1 = line) |
| incremental value | preference i produces a response the other does not | controllability: S(i⁺) − S(j⁺) response vectors, and whether S_j(j⁺) beats S_j(i⁺) |

S_i are higher-is-better semantic scores. The measurements use the
switch-controlled, steady-window protocol (V4-C2R/C2F).

**Decision rule.** A pair is called redundant only when all of these agree:
high ρ, low off-diagonal mass, PC1 near 1, **and** no incremental
controllability for one member (its own preference does not beat the other
preference on its own score). PC1 or correlation alone never drops an
objective, because they carry no semantic meaning. Of a redundant pair, the
member that has no incremental controllability is dropped, not blended.
Blending would add a coefficient with no evidence that it carries new
information.

Cross-objective improvement is **synergy, not redundancy**. Two relevant
objectives may help each other. What matters is that their response profiles
differ (see [MORL semantic criteria](../../contracts/teacher_v4/teacher-v4-c3-evaluation-contract.md)).

## Stability of the relation across policies and regimes (Layer 4, PLANNED)

A relevance–redundancy measurement is a property of the pair **and** of the
policy and regime it was measured under: Redundancy(r_i, r_j) =
f(regime, policy). A policy trained on an objective set can itself make two
objectives co-vary, so redundancy measured on one MORL policy may be an
artifact of that policy. The method therefore has four layers:

| layer | question | measures |
|---|---|---|
| 1 local association | do the outcomes co-vary? | Spearman ρ, off-diagonal quadrant mass |
| 2 outcome geometry | is the outcome cloud lower-dimensional? | PC1 share, intrinsic dimensionality |
| 3 controllability | does preference i produce behavior distinct from j? | incremental controllability (above) |
| 4 stability | does the relation hold when the policy or regime changes? | layers 1–3 repeated per policy / regime |

Layer 4 varies: policy seeds, policy checkpoints (early/late), policy type
(MORL, single-objective experts, perturbed policies), plant bins, terrain
regimes, simulator/domain.

- **Screening order.** Layers 1–2 are cheap and run on every pair in every
  regime. Layer 3 needs training and runs only on pairs that layers 1–2 flag.
- **Pre-registered stability rule.** Before any Layer 4 measurement, a
  contract fixes the regimes, the minimum support per regime, and the rule.
  Default: a pair is *robustly redundant* only if every regime with enough
  support passes the redundancy decision rule above. Any allowed fraction of
  failing regimes must be fixed in advance, never after seeing results.
- **Selection stays offline.** Groupings are never merged or split online. A
  changed grouping needs a new selection round, a new frozen objective set and
  new training.

**Evidence status.** Current evidence covers policy seeds and checkpoints only
(10 V4-C checkpoints and 6 V4-C2S-R1 checkpoints, one substrate). No
cross-plant, terrain or simulator claim is made. The first planned step costs
no training: split existing rollouts into plant bins by the privileged e_t
(mass, friction under stock DR) and recompute layers 1–2 for A–S per bin.

Method statement: objective selection retains semantic factors that provide
incremental controllability and whose relevance–redundancy structure remains
stable across policies and operating regimes.

## Selection pipeline, revised after the standing finding (PLANNED)

The V4-C selection round kept A and O as distinct features. The frozen set
{T, D, O} still made standing the objective's own optimum over most of the
simplex ([substrate attribution](../../verdicts/teacher_v4/teacher-v4-c3-substrate-attribution-verdict.md)).
Feature quality therefore does not imply set viability. The pipeline gains a
role screen, a task-conditional layer and a joint-viability layer:

    raw reward library
      1. task-role screen       objective candidate | constraint | regularizer | feasibility / gait prior
      2. feature selection      relevance, redundancy, incremental controllability (layers 1-3 above)
      3. conditional selection  layers 1-3 repeated inside the task-valid (locomoting) regime
      4. grouping               raw terms -> behavioral objectives, fixed α
      5. joint viability        does wᵀR keep locomotion optimal over the intended preference support? (F1)
      6. freeze                 objective set, then MORL training and evaluation

- **Roles decide where a term enters.** Objective candidates get a
  preference weight. Constraints, regularizers and feasibility priors are
  candidates for a preference-invariant term R_shared, never inside an
  objective. Today non-objective terms have zero training weight
  ([raw reward library](../../../talon_rl/rewards/README.md#raw-reward-library-v4-teacher)).
- **Raw vs task-conditional relevance.** A signal can be easy to control
  through a trivial mechanism. `ang_vel_xy_l2` is best when the robot
  stands. Layer 3 asks whether the signal still varies and is controllable
  while the task stays solved: Var(r | locomoting), not Var(r).
- **Layer 5 is a set property.** Two features can each be relevant and
  mutually non-redundant, yet together make a trivial behavior optimal.
- **Data requirement.** Layer 3 needs a policy bank that locomotes in
  diverse ways. The current bank has M0 (one policy) and three partial
  walkers, not enough for policy-independent conditional statistics.
- **Leakage.** The V4-C/C2/C3 checkpoints and evaluation outputs are
  selection data for the next round. They are never evaluation data for
  the objective set that round produces.

**Working design (2026-09-29).** Tracking leaves the preference simplex and
becomes the locomotion task requirement. The preferences are R / V / O inside
feasible locomotion. See [behavior–preference hierarchy](behavior-preference-hierarchy.md).

## Separation of selection and evaluation (no leakage)

    selection data (policy bank)           frozen objective set          independent test
    ────────────────────────────           ────────────────────          ────────────────
    V4-C checkpoints (T/A/O/S)     ──►     {T, A, O}             ──►     V4-C3: new training runs,
    V4-C2S-R1 checkpoints (S1)             (contract 7d7a087)            new checkpoints, evaluation
    anchor (seen) sets only                                              contract frozen before running

- Selection used only the full-set anchor (a cardinality seen in training)
  of already-trained policies. No held-out-cardinality result was used.
- The chosen set was frozen before V4-C3 training. V4-C3 trains from scratch
  and is evaluated under its own preregistered contract. Its results are not
  used to re-select objectives. A later change to the objective set needs a
  new selection round and a new training/evaluation round.

## Result on this task (V4-C selection round)

Relevance–redundancy matrix (10 primary V4-C checkpoints, medians; ρ per step
/ off-diagonal / PC1). Source:
[V4-C2F verdict](../../verdicts/teacher_v4/teacher-v4-c2f-specificity-audit-verdict.md).

|   | T | A | O |
|---|---|---|---|
| A | −0.04 / 0.52 / 0.56 | | |
| O | −0.06 / 0.51 / 0.53 | 0.16 / 0.46 / 0.68 | |
| S | −0.05 / 0.52 / 0.56 | **0.84 / 0.14 / 0.96** | 0.21 / 0.44 / 0.61 |

Incremental controllability for the only high-redundancy pair: AA − AS is
positive in 9/10 checkpoints, SS − SA in only 4/10. With S retrained as action
jerk ([V4-C2S-R1](../../verdicts/teacher_v4/teacher-v4-c2s-r1-action-jerk-verdict.md)),
the A–S1 pair gave ρ 0.80, PC1 0.94, SS − SA positive in 0/6 and AA − AS in 6/6.
Other smoothness forms were measured too
([V4-C2S](../../verdicts/teacher_v4/teacher-v4-c2s-smoothness-verdict.md)).

Clusters: **T independent; O independent; {A, S} one behavioral cluster, in
which A has incremental controllability and S does not.**

Selected objective set: **{T, O, D}**, where D is Dynamic Stability. Current
instantiation:

    T  = track_lin_vel_xy_exp + track_ang_vel_z_exp   (stock weights)
    O  = flat_orientation_l2
    D  = 1.0 · r̃_A (ang_vel_xy_l2) + 0.0 · r̃_S

Smoothness is a candidate constituent of D. The audit found it adds no
incremental controllability on this substrate, so its coefficient is zero in
the current formulation. A non-zero α_S needs its own contract, with α fixed
before training and a paired comparison against α_S = 0. Code and artifacts
keep the label A for D.
