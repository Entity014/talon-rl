# Preference-to-Behavior Ordering Audit Contract

Status: **PREDECLARED — MEASUREMENT ONLY, NO TRAINING**

Date: 2026-09-24

## Scientific question

Is the observed winner rotation better characterized as degradation of the learned mapping

    preference w -> behavioral trade-off

rather than as degradation of a scalar return or scalar semantic score?

The desired MORL property is not that the center/uniform preference is simultaneously best on all objectives. The desired property is:

> increasing preference weight on objective i should move behavior monotonically in the semantic direction of objective i, while allowing trade-offs in other objectives.

## Frozen controller and evaluation foundation

- V2-B single-site FiLM actor
- Foundation V2
- GAE lambda=.95
- repaired transformed-action PPO semantics
- unchanged critic / optimizer / architecture
- 8 environments
- 64 rollout steps
- 4 matched reset suites, seeds 840001..840004
- same normalized objective definitions and physical proxies

No optimizer step is allowed.

## Strict held-out policy corpus

Use exactly the 3 independent weighted-PPO CONTROL paths previously used for held-out validation:

- train seed 980001: base -> u1 -> ... -> u8
- train seed 981001: base -> u1 -> ... -> u8
- train seed 982001: base -> u1 -> ... -> u8

Shared base:
`runs/v2b_adam_continuous-2026-09-24/model_75.pt`

This gives:
- 25 unique checkpoints
- 24 finite transitions
- 96 axis-transition samples

No discovery-path checkpoint is included.

## Frozen preference ladder

For each semantic axis i in {T,A,O,S}, evaluate five points:

    q = w_i in {0.10, 0.25, 0.40, 0.55, 0.70}

with all other weights equal:

    w_j = (1-q)/3,  j != i

Thus:
- q=.10 = away-from-i
- q=.25 = center
- q=.40 = intermediate-1
- q=.55 = intermediate-2
- q=.70 = i-heavy

No ladder point is changed after observing results.

## Per-point behavioral signals

For axis i and each suite:

### Objective response

    b_obj_i(q) = mean normalized objective-i return

Higher is better.

### Physical response

Higher-is-better physical score:

    T: -tracking_error
    A: -ang_vel_xy
    O: -tilt_deg
    S: -action_rate

Call this:

    b_phys_i(q)

The audit does not combine objective and physical signals into a new reward.

## Ordering metrics at each checkpoint / axis

All metrics are computed separately for objective and physical behavior and then reported jointly.

### 1. Pairwise ordering accuracy

Across all C(5,2)=10 preference pairs q_a > q_b:

    correct iff b(q_a) > b(q_b)

Report fraction correct across pairs and suites.

### 2. Interior pairwise ordering accuracy — PRIMARY

To avoid reducing the metric to the same heavy-vs-center comparison already used by the semantic gate, exclude the direct pair (0.70,0.25).

Primary ordering accuracy therefore uses the remaining 9 pairs.

### 3. Adjacent monotonic accuracy

For adjacent ladder pairs:

    .10 -> .25 -> .40 -> .55 -> .70

report fraction with positive behavioral increment.

### 4. Preference-response rank correlation

Per suite:

    Spearman(q, b(q))

Average across the 4 suites.

### 5. Intermediate-between-endpoints fraction

For q in {.25,.40,.55}, report the fraction whose behavior lies between the q=.10 and q=.70 endpoint behaviors.

### 6. Reset consistency

For each of the 9 primary preference pairs, report fraction of the 4 reset suites with correct ordering.

Checkpoint-level reset-consistent pair fraction:

    fraction of pairs correct in at least 3/4 suites

### 7. Joint objective-physical ordering

For each primary pair and suite, count correct only if BOTH objective and physical behavior move in the intended direction.

Report:
- joint primary pairwise accuracy
- joint reset-consistent pair fraction

No scalar averaging between objective and physical channels is used for the primary joint metric.

## Transition-level forgetting quantities

For each consecutive checkpoint transition and axis compute change in:
- objective primary ordering accuracy
- physical primary ordering accuracy
- joint primary ordering accuracy
- objective rank correlation
- physical rank correlation
- objective reset-consistent pair fraction
- physical reset-consistent pair fraction
- joint reset-consistent pair fraction

Target events reuse the frozen semantic endpoint labels:
- PASS -> FAIL
- FAIL -> PASS
- semantic-score change

## Baseline controls

Compare ordering deterioration against previously validated scalar descriptors from the same held-out corpus:

1. Delta heavy-only J_i
2. Delta mean objective heavy-vs-center relation
3. Delta mean physical heavy-vs-center relation

The ordering hypothesis must provide information beyond these scalar controls.

## Primary forgetting hypothesis

Winner rotation is considered to involve **preference-map ordering collapse** only if all hold:

1. At least 75% of PASS -> FAIL events show deterioration in joint primary ordering accuracy OR joint reset-consistent pair fraction.
2. This sensitivity exceeds deterioration of the strongest scalar mean-relation control by at least 0.10.
3. At least 25% of PASS -> FAIL events show ordering deterioration while BOTH mean objective and mean physical heavy-center relations are non-decreasing, demonstrating information beyond scalar relation.
4. Across all 96 axis-transitions, change in joint primary ordering accuracy has |Spearman| >=0.50 with semantic-score change.
5. Ordering-change direction is qualitatively consistent across all 3 training seeds.
6. FAIL -> PASS events show ordering improvement in at least 60% of events when at least 4 such events exist.

## Prospective secondary check

For source checkpoints that are PASS, test whether source ordering robustness predicts next-checkpoint retention beyond source mean relation.

This is secondary and descriptive only because prior prospective context analysis was underpowered.

Report:
- retention rate for source joint primary ordering accuracy >=0.75 vs <0.75
- same split within upper/lower halves of source mean-relation strength when sample counts permit

No model fitting or threshold tuning is performed; 0.75 is frozen a priori.

## Decision

### ORDERING HYPOTHESIS SUPPORTED
Preference-to-behavior map degradation explains semantic forgetting beyond mean relation. This authorizes a separate pairwise/ranking-target design audit, not training.

### ORDERING INFORMATIVE BUT NOT INCREMENTAL
Ordering tracks semantics but does not add enough information beyond heavy-center mean relation. No ranking objective is authorized.

### ORDERING NOT SUPPORTED
The preference map remains ordered despite semantic forgetting, or ordering changes do not generalize. Close the ranking branch.

No pairwise loss, margin delta, ranking coefficient, reward, optimizer, architecture, or training run is authorized in this audit.
