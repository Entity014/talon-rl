# V3-G1 Variable-Cardinality Generalization Contract

Status: **PREDECLARED — FROZEN BEFORE G1 TRAINING**
Date: 2026-09-25

## Question

Can one objective-set-conditioned controller, with one parameterization and no cardinality-specific modules, train on selected active-set cardinalities and generalize zero-shot to a cardinality never exposed anywhere in training?

G1 isolates **cardinality generalization** only.

It does not claim:
- unseen combination generalization (G2);
- unseen objective identity generalization (G3);
- Smoothness repair.

## Frozen start

Both folds initialize from the exact G0 representation-equivalent model:

    runs/objective_set_g0_structural_parity-2026-09-25/objective_set_g0_init.pt

Architecture and token dimension are identical across folds.

No architecture change after G0 is authorized.

## Folds

G1-2:
    training cardinalities = {3,4}
    held-out cardinality   = 2

G1-3:
    training cardinalities = {2,4}
    held-out cardinality   = 3

The held-out cardinality is forbidden from every training-time path.
## Leakage prohibition

For a fold, held-out cardinality must not appear in:
- PPO rollouts;
- critic targets or support;
- edge-retention/reference pairs;
- replay or auxiliary support;
- curriculum;
- checkpoint selection;
- engineering eligibility checks used to choose a checkpoint;
- semantic evaluation used during training.

Technical ability of the G0 teacher to evaluate the held-out cardinality is irrelevant and must not be used.

A single leakage event invalidates that fold.

## Objective-set sampler

Vocabulary is T/A/O/S.

At every training update:
1. sample an allowed active-set cardinality uniformly;
2. sample an active subset uniformly among all subsets of that cardinality;
3. for each environment lane, sample either:
   - center with probability 0.20;
   - one active-objective heavy endpoint with probability 0.40;
   - Dirichlet(1,...,1) interior preference with probability 0.40.

Heavy endpoint:
    target objective weight = 0.70
    remaining 0.30 divided uniformly among other active objectives.

No inactive objective token is present.
No inactive objective value is queried.
No inactive objective reward contributes to scalarization.
## Training budget and seeds

Frozen budget per fold/seed:

    updates              = 30
    rollout horizon      = 32
    num environments     = 8
    gamma                = 0.99
    GAE lambda           = 0.95
    actor learning rate  = 1e-3
    critic learning rate = 1e-3
    PPO clip epsilon     = 0.20
    gradient clip norm   = 1.0

Independent training seeds:

    73101
    73102
    73103

All six runs are required for final G1 characterization.

No poor seed may be replaced.

## Token-query critic training

For each rollout with active token set O of size m:

    V_i = V(s,e_i,z_O),  i in active set only

Targets are vector GAE/returns for active objectives only.

Critic loss:

    L_V = mean_i (V_i - R_i)^2

The shared token-query critic body/value basis is trained directly through this API.

Forbidden:
- querying inactive tokens during critic training;
- constructing a hidden four-head target and masking afterward;
- objective-specific critic heads.
## Actor objective

For each transition, scalarized PPO advantage uses only active objectives:

    A_w = m * sum_i w_i A_i

where i ranges only over the active set.

The multiplicative m preserves the mean scale convention used by the Phase-1 four-objective late-weighted PPO.

Action/log-prob semantics remain the repaired tanh-squashed PPO contract.

## Active-set-local edge retention

Authority durability is generalized without fixed T/A/O/S edge tables.

For each training-supported active set O:
- define center preference over O;
- define heavy endpoint for every active objective;
- compute all unordered heavy-heavy and heavy-center action distances inside O only.

For a matched state s:

    d_ij(s,O) = ||pi(s,O_i)-pi(s,O_j)||_2

Reference distances come from the frozen G0 model evaluated only on training-supported cardinalities.

Asymmetric floor:

    L_edge = E[max(0, gamma_edge d_ref - d_current)^2]

Frozen:
    gamma_edge = 0.90
    rho         = 0.25

Bound retention-gradient contribution by the validated norm budget:

    alpha = min(beta0, rho ||g_base||/(||g_edge||+eps))

with:
    beta0 = 2.497041993384243

No edge from held-out cardinality may be referenced.
## Checkpoint policy

Snapshots are saved at:

    u0, u10, u20, u30

Checkpoint selection for final evaluation is fixed in advance:

    use u30

No semantic score, held-out-cardinality metric, or post-hoc engineering result may select among u10/u20/u30.

Earlier snapshots are diagnostics only.

## Seen-cardinality retention

At u30, evaluate every training-seen cardinality.

Required engineering interpretation:
- permutation hard gate remains exact within 1e-6;
- finite action/log-prob/value outputs;
- PPO ratio contract maintained;
- no architecture/cardinality-specific parameter changes.

Semantic evaluation uses the same standardized protocol as Phase 2 contract.

T/A/O remain required whenever present.
S is reported but not required.

## Held-out-cardinality evaluation

Primary G1 result.

G1-2:
    evaluate all six m=2 active sets.

G1-3:
    evaluate all four m=3 active sets.

For each set:
- endpoint semantics;
- center compromise;
- every pairwise heavy-heavy edge;
- frozen interior samples;
- preference authority;
- token-query critic validity;
- survival/robustness.

No held-out set may influence training or checkpoint choice.
## Full-set anchor

m=4 is present in both folds and is evaluated at u30 as the common anchor.

Require:
- T endpoint semantic PASS;
- A endpoint semantic PASS;
- O endpoint semantic PASS;
- S reported exactly;
- center/continuum/critic/robustness reported;
- no objective-specific repair.

This anchor detects whether generalized training destroys the Phase-1-valid semantics.

## Fold-level success

A fold passes cardinality generalization only if:

1. no held-out-cardinality leakage occurred;
2. one unchanged architecture executed the held-out cardinality;
3. permutation hard gate PASS;
4. engineering validity (authority, critic, robustness, PPO consistency) is valid on held-out sets;
5. T/A/O semantic endpoints pass whenever present, aggregated using the frozen per-set protocol;
6. full-set m=4 anchor retains T/A/O validity.

S is descriptive and does not determine G1 PASS.

## Aggregate G1 verdict

G1 is supported only if both G1-2 and G1-3 pass on the preregistered three-seed characterization.

Single-seed results are screens, not the final thesis claim.

If either fold systematically fails:
- report cardinality generalization as not established;
- do not reinterpret as unseen-combination failure;
- do not open G2;
- do not patch a specific objective/cardinality inside G1.
## Stop rules

Immediate invalidation:
- held-out-cardinality leakage;
- cardinality-specific architecture change;
- objective-specific branch/head/loss;
- reward redefinition.

Stop before G2 if:
- held-out cardinality cannot be executed with same parameters;
- broad authority/critic/robustness failure occurs;
- T/A/O semantic validity systematically degrades.

G2 is authorized only after the complete three-seed G1 verdict passes.

## Frozen implementation invariants

Training code must log, for every update:
- sampled active cardinality;
- sampled active objective IDs per lane;
- sampled weights;
- queried critic token IDs;
- retention active-set IDs;
- assertion that all cardinalities belong to fold training support.

A provenance manifest must hash:
- this contract;
- G0 init;
- training script;
- objective-set model source;
- fold specification;
- seed and budget.

## Evaluation addendum — authority operationalization (frozen before G1 evaluation)

To operationalize the already-required preference-authority gate without using semantic outcomes, compare each u30 model against the frozen G0 initialization on the same fixed probe states and active set.

For each evaluated active set:
- compute mean pairwise action distance across center + all heavy endpoints;
- compute mean Frobenius norm of the action Jacobian on an orthonormal basis of the active-set simplex tangent space at center.

Require:

    pairwise authority retention >= 0.75
    tangent Jacobian retention    >= 0.75

relative to G0 for the identical active set.

The 0.75 retention threshold is inherited from the Phase-1 authority-durability gate and is fixed before G1 semantic evaluation.

Fresh critic validity is evaluated on active-token H32 returns only:

    mean EV > 0
    negative EV fraction <= 0.25

Permutation tolerance remains 1e-6.
