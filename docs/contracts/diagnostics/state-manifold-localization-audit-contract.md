# State-Manifold Localization Audit Contract

Status: **PREDECLARED — READ-ONLY, NO TRAINING / NO INTERVENTION DESIGN**

Date: 2026-09-24

## Question

Do collapse-producing finite updates rewrite a reproducible **region of the actual task-relevant source-state manifold** more strongly than retained/non-robust updates, even though global update summaries overlap?

The goal is localization of a region, not discovery of another scalar predictor.

## Primary corpus

Use exactly the independent v18 source-PASS transitions:

    seeds 983001, 984001, 985001
    eligible transitions       = 22
    robust semantic collapse   = 10
    retained / non-robust      = 12

Robust-collapse labels remain frozen from v18. Boundary-sensitive failures are not positive cases.

## State source

For each transition-axis, use the **source-policy / source-visitation** heavy-preference states already cached in the v20 state cache:

    4 reset suites x 64 steps x 8 envs = 2048 states per transition

No target states enter the manifold partition.
No simulator rerun is authorized.

Observation layout is frozen as:

    0:3    base linear velocity
    3:6    base angular velocity
    6:9    projected gravity
    9:12   command
    12:24  joint position
    24:36  joint velocity
    36:48  previous action

## State-wise functional rewrite

On every source state s, evaluate source and target policies under:
- axis-heavy preference;
- center preference.

Define state-wise metrics:

### Action rewrite

    A_center(s) = ||a_target(s,wC) - a_source(s,wC)||_2
    A_heavy(s)  = ||a_target(s,wH) - a_source(s,wH)||_2

### Heavy-vs-center response rewrite

    R_source(s) = a_source(s,wH) - a_source(s,wC)
    R_target(s) = a_target(s,wH) - a_target(s,wC)

    R_change(s)   = ||R_target(s)-R_source(s)||_2
    R_rotation(s) = 1 - cos(R_target(s),R_source(s))

### Preference-Jacobian rewrite

At center preference:

    J_source(s) = partial a_source / partial w
    J_target(s) = partial a_target / partial w

    J_change_rel(s) = ||J_target-J_source||_F / (||J_source||_F + eps)
    J_rotation(s)   = 1 - cos(vec(J_target),vec(J_source))

Primary localization families:

    R_change
    R_rotation
    J_change_rel
    J_rotation

Action rewrite is secondary/contextual.

## Label-blind manifold partition

Pool source states from all 22 transitions, without using collapse labels.

1. Standardize each of the 48 observation dimensions using pooled source-state mean/std.
2. PCA fitted label-blind on the standardized states.
3. Retain the smallest component count explaining >=90% variance, capped at 16 components and floored at 4.
4. K-means on PCA coordinates:

       K = 16
       random_state = 260924
       n_init = 20

No alternative K, PCA threshold, embedding, or clustering algorithm may be chosen after seeing outcome labels.

## Transition-level regional summaries

States are not treated as independent statistical samples.

For each transition and cluster c report:

    occupancy_c = fraction of its 2048 source states assigned to c

For each functional rewrite metric X:

    mean_X_c = mean X over transition states in c
    p90_X_c  = 90th percentile X over transition states in c
    mass_X_c = occupancy_c * mean_X_c

A transition-cluster entry is eligible only if that transition contributes at least 16 states to the cluster.

## Frozen region-localization gate

A cluster is a **reproducible collapse-localized region** for a primary metric only if all hold:

1. eligible in >=8/10 robust-collapse transitions;
2. eligible in >=8/12 retained/non-robust transitions;
3. transition-level ROC AUC of `mean_X_c` for robust collapse >= .75;
4. robust median `mean_X_c` exceeds retained median by >=0.5 retained IQR;
5. >=8/10 robust transitions exceed the retained median among eligible entries;
6. direction is consistent across all 3 seeds when both classes have eligible entries;
7. regional AUC exceeds the corresponding **global transition mean** metric AUC by >= .05;
8. the same cluster also has AUC >= .70 for either `p90_X_c` or `mass_X_c`.

No cluster may be selected only because one phase looks favorable.

## Reproducibility across state context

For every cluster passing or approaching the regional gate, report label-blind context descriptors:

- phase occupancy Q1/Q2/Q3/Q4;
- base linear speed median;
- base angular speed median;
- projected-gravity XY norm median (tilt proxy);
- command linear/turn components median;
- joint-position norm median;
- joint-velocity norm median;
- previous-action norm median.

These descriptors interpret a region; they do not define it.

## Phase localization

After cluster identities are fixed label-blind, report each cluster's rewrite profile over Q1-Q4.

A phase pattern may be called **phase-concentrated** only if the same cluster/metric already passes items 1–6 above on pooled states and one quarter contains >=40% of that cluster's total rewrite mass in robust collapses.

Phase analysis cannot create a new region candidate by itself.

## Global controls

For each transition compute global means over all 2048 source states for every functional rewrite metric.

This audit asks whether localization adds information beyond these global summaries.

Also report state-manifold occupancy shift between robust and retained source states, but occupancy alone is not a collapse mechanism because clustering is source-state-only.

## Verdicts

### REPRODUCIBLE LOCALIZED REWRITE REGION
At least one cluster and one primary functional metric passes all eight localization criteria.

### DISTRIBUTED / WEAK LOCALIZATION
No cluster passes all criteria, but one or more clusters reach AUC >=.70 with seed-consistent direction and improve >=.03 over the global metric.

### NO REPRODUCIBLE LOCALIZATION
No cluster provides stable improvement over global functional-change summaries.

## Decision scope

This audit does NOT authorize:
- local trust regions;
- region-conditioned rehearsal;
- state-weighted PPO;
- manifold-aware regularization;
- architecture changes;
- any new training run.

Any intervention requires a separate prospective/causal validation after localization is independently reproduced.
