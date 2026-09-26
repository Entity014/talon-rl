# State-Manifold Localization Audit Verdict

Status: **FROZEN — DISTRIBUTED / WEAK LOCALIZATION; NO REPRODUCIBLE COLLAPSE REGION**

Date: 2026-09-24

## Question

Do collapse-producing updates rewrite a reproducible region of the actual source closed-loop state manifold more strongly than retained updates, despite weak global functional discrimination?

## Corpus and manifold construction

The audit uses the independent v18 source-PASS corpus:

    transitions             22
    robust collapse         10
    retained/non-robust     12

For each transition, 2048 actual source-policy/source-visitation states were used:

    4 reset suites x 64 steps x 8 envs

The manifold partition was fit without collapse labels on the pooled 48-D source observations:

    standardize 48-D observations
    PCA, 90% target with 16-component cap
    K-means K=16, n_init=20, random_state=260924

The 16-component cap retained:

    79.87% of standardized observation variance

Thus the partition is intentionally compact but does not capture the full 90% target because the predeclared dimensionality cap became active.

## Global state-wise rewrite controls

Transition-level global means remain weak/moderate discriminators:

    heavy-center response change       AUC 0.450
    heavy-center response rotation     AUC 0.633
    preference-Jacobian relative change AUC 0.583
    preference-Jacobian rotation       AUC 0.608
    center action rewrite              AUC 0.592
    heavy action rewrite               AUC 0.592

This reproduces the v19/v20 picture: global function-space change overlaps strongly between collapse and retained updates.

## Frozen localization gate

No cluster/primary-metric pair satisfies all eight predeclared localization criteria.

Formal verdict:

> **DISTRIBUTED / WEAK LOCALIZATION**

There is no reproducible localized rewrite region strong enough to authorize a region-conditioned intervention.

## Strongest localized near-signal: cluster 2 Jacobian change

The strongest regional result is:

    cluster 2
    metric: preference-Jacobian relative change

Transition-level result:

    regional mean AUC        0.717
    global metric AUC        0.583
    localization gain       +0.133

    robust median            0.04724
    retained median          0.04101
    retained IQR             0.00716
    effect                   0.87 retained-IQR units

    robust above retained median  8/10
    seed direction consistency     3/3

This is a meaningful improvement over the global summary.

However, the region fails the frozen localization gate because:

    required regional AUC >= .75       FAIL: .717
    required p90 or mass AUC >= .70    FAIL: .642 / .683

Therefore the signal is not strong enough to call a reproducible collapse-localized region.

## Cluster 2 visitation is not collapse-specific

Cluster occupancy is essentially identical:

    robust median occupancy    5.79%
    retained median occupancy  5.71%
    occupancy AUC              0.554

Thus collapse policies do not simply visit this region more often.

The near-signal concerns **how strongly the update changes preference sensitivity inside the region**, not whether the region exists only in collapse trajectories.

## State-context interpretation of cluster 2

Label-blind cluster context:

    phase occupancy:
      Q1   0.0%
      Q2  24.9%
      Q3  38.3%
      Q4  36.8%

    median base linear speed norm     0.744
    median base angular speed norm    1.017
    median gravity-XY norm            0.062
    median joint-position norm        1.040
    median joint-velocity norm       11.581
    median previous-action norm       3.238

Relative to the 16 learned regions, cluster 2 is on the higher side for linear speed and joint velocity, but is not an extreme angular-rate or tilt region.

It therefore resembles a **later-rollout, dynamically active locomotion region**, rather than a simple high-tilt/failure state cluster.

## Phase distribution of rewrite inside cluster 2

For the near-signal `J_change_rel`, robust-collapse rewrite mass is distributed as:

    Q1   0.0%
    Q2  24.8%
    Q3  39.8%
    Q4  35.4%

The frozen phase-concentration threshold is >=40% in one quarter.

Observed maximum:

    39.8% in Q3

Therefore Jacobian rewrite in this region is **not formally phase-concentrated**, although it is clearly restricted to mid/late rollout because the region itself is absent from Q1.

## Axis heterogeneity

Cluster-2 Jacobian change is not a universal semantic-axis mechanism.

Median regional `J_change_rel`:

### Angular

    robust n=3    median 0.05965
    retained n=5  median 0.04278

Direction strongly matches the pooled near-signal.

### Orientation

    robust n=2    median 0.04166
    retained n=2  median 0.03800

Direction matches weakly.

### Smoothness

    robust n=5    median 0.04620
    retained n=4  median 0.05311

Direction reverses.

### Tracking

No robust Tracking event exists in this independent corpus.

Thus the pooled 3-seed consistency does not imply cross-objective universality. The strongest localization appears driven primarily by Angular, with weaker Orientation support and contradictory Smoothness behavior.

This materially limits causal interpretation.

## Other regions

No other cluster reaches the frozen weak-localization rule.

The next strongest regional AUCs are approximately:

    cluster 1  response rotation      0.683
    cluster 12 response rotation      0.683
    cluster 5  Jacobian rotation      0.675
    cluster 10 response rotation      0.675

None provide a stable, strong improvement over their corresponding global functional metric.

## Scientific interpretation

v20 suggested that task-relevant source visitation might reveal update effects hidden by generic probes. The present state-wise localization audit provides partial support for that intuition but not a clean mechanism.

A particular later-rollout source-state region improves preference-Jacobian discrimination from global AUC 0.583 to regional AUC 0.717, with 3/3 seed-consistent direction. However:

- the effect misses the predeclared AUC threshold;
- tail/mass summaries do not confirm it;
- cluster occupancy is not collapse-specific;
- the effect is not cross-axis consistent;
- no second region reproduces an equally strong pattern.

Therefore:

> **Collapse-producing functional rewrite is somewhat more localized than global summaries suggest, but the localization is weak, axis-dependent, and insufficiently reproducible to identify a shared state-manifold failure region.**

This is consistent with a distributed/high-dimensional failure rather than a single compact region that can be safely targeted.

## Decision

No local trust region, region-conditioned rehearsal, state-weighted PPO, or manifold-aware penalty is authorized.

The state-manifold localization branch is closed under the current evidence.

If controller research continues beyond the thesis-critical path, the remaining evidence favors a formulation-level alternative—such as explicit policy-family / Pareto-set construction—over continued post-hoc search for a single scalar or local region in one evolving shared policy.

## Thesis wording

> State-wise localization on the actual source visitation manifold did not reveal a reproducible collapse-specific rewrite region. A label-blind 16-region partition identified one later-rollout region in which preference-Jacobian change improved discrimination from global AUC 0.583 to 0.717 with consistent direction across three seeds, but the effect failed the predeclared regional threshold, was not confirmed by tail or rewrite-mass metrics, and reversed for Smoothness. The region was also visited equally often in retained and collapsing transitions. These results suggest that collapse-producing functional changes are distributed and axis-dependent rather than concentrated in a single reusable state-space region.
