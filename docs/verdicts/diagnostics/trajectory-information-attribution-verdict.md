# Trajectory Information Attribution Audit Verdict

Status: **FROZEN — FORMAL FAMILY GATE INCONCLUSIVE; RELATIONAL TEMPORAL/PERSISTENCE SIGNAL STRONGLY IDENTIFIED; REWARD REDESIGN NOT AUTHORIZED**

Date: 2026-09-24

## Question

What trajectory information does the frozen semantic endpoint use that the current scalar heavy-preference objective return discards?

This audit follows the trajectory-level sufficiency failure and is measurement-only.

## Protocol

The exact four policy paths and matched-reset endpoint suites were replayed:
- V2-B lambda=.95
- V2-B lambda=1.00
- paired weighted-PPO control
- paired max-min floor

Across 19 unique checkpoints, the audit re-ran the frozen 64-step endpoint trajectories with the same:
- 8 environments
- 4 matched suites
- seeds 840001..840004
- T/A/O/S-heavy and center preferences
- objective normalization and physical proxies

No optimizer step and no reward change occurred.

For every axis the audit recorded per-step:

    objective under heavy preference
    objective under matched center
    physical proxy under heavy preference
    physical proxy under matched center

and formed higher-is-better relational signals:

    objective advantage(t) = heavy objective - center objective
    physical advantage(t)  = center physical - heavy physical

Frozen descriptor families covered:
- aggregate
- temporal early/mid/late structure
- tail / worst-window statistics
- relational persistence

## Formal predeclared family verdict

The automated family gate returned:

> **NO SIMPLE TRAJECTORY FAMILY DOMINATES**

However, the reason requires an explicit methodological qualification.

The predeclared aggregate family included:
- mean objective heavy-center margin
- mean physical heavy-center margin

These quantities are mathematically the same matched semantic mean margins used as continuous attribution targets. Their change therefore correlates with the target at approximately rho=1 by construction.

Consequently the aggregate-family median becomes approximately 1.0, making the predeclared requirement that another family exceed the aggregate median by +0.10 impossible.

This does **not** invalidate the collected trajectory evidence. It means the family-level winner criterion was poorly specified for distinguishing richer temporal information from the scalar training return.

The formal family gate is therefore frozen as inconclusive and is not retroactively changed.

## Scalar training-return baseline

The relevant current training summary is the heavy-policy scalar objective return:

    J_i = mean normalized objective-i return under i-heavy preference

Its change has only weak rank alignment with the semantic objective-margin change:

- Spearman rho = **0.251**
- sign agreement = **0.578**

This reproduces the earlier trajectory-sufficiency diagnosis.

## Strongest non-tautological attribution signals

When compared descriptively against the actual scalar-return baseline, the most informative descriptors are relational temporal and persistence quantities.

### Late trajectory relation

| Descriptor | Target | Spearman rho | Sign agreement |
|---|---|---:|---:|
| late objective advantage | Delta semantic objective margin | **0.872** | 0.734 |
| late physical advantage | Delta semantic physical margin | **0.874** | **0.828** |
| physical late-minus-early advantage | Delta semantic physical margin | **0.882** | 0.797 |
| objective late-minus-early advantage | Delta semantic objective margin | **0.842** | 0.750 |
| physical advantage slope | Delta semantic physical margin | **0.863** | 0.797 |
| objective advantage slope | Delta semantic objective margin | **0.810** | 0.766 |

The direction of association is consistent across all four policy paths and all four axes for these key descriptors.

### Persistence / final-window relation

- final-16 objective advantage: rho = **0.776**
- final-16 physical advantage: rho = **0.791**
- objective positive-advantage fraction: rho = **0.722**
- physical positive-advantage fraction: rho = **0.726**
- objective minimum-prefix mean: rho = **0.621**
- physical minimum-prefix mean: rho = **0.595**

Again, the qualitative relation is consistent across all four paths and all four axes for the strongest descriptors.

### Worst-window relation

- worst-16 objective advantage: rho = **0.752**
- worst-16 physical advantage: rho = **0.695**

These are informative, but the broader tail family is mixed: absolute heavy-trajectory variance and p90/p95/max statistics are much weaker than the matched relational worst-window descriptors.

This distinction matters:

> The evidence favors **heavy-vs-center relational temporal structure**, not generic risk/tail statistics of the heavy trajectory alone.

## PASS -> FAIL attribution

There are 9 endpoint PASS -> FAIL transitions in the audited paths.

Among these transitions:
- late objective advantage decreases in **9/9**
- late physical advantage decreases in **9/9**
- objective late-minus-early advantage decreases in **8/9**
- physical late-minus-early advantage decreases in **8/9**
- objective advantage slope decreases in **8/9**
- physical advantage slope decreases in **8/9**
- final-16 objective advantage decreases in **8/9**
- final-16 physical advantage decreases in **8/9**
- worst-16 objective advantage decreases in **8/9**
- worst-16 physical advantage decreases in **8/9**

By contrast, the heavy-only scalar training return decreases in only **6/9** PASS -> FAIL events.

This is the clearest attribution result in the audit.

## Example: Orientation, lambda=.95, u10 -> u25

The scalar training objective improves:

    Delta J_O = +0.003153

while semantic competence is lost.

At the same transition:

    Delta worst-16 objective advantage = -0.000776
    Delta worst-16 physical advantage  = -0.1606
    Delta objective positive fraction  = -0.0938
    Delta physical positive fraction   = -0.0820
    Delta objective late-minus-early   = -0.00182
    Delta physical late-minus-early    = -0.1963

Thus the scalar objective says "better," while the matched trajectory relation becomes less persistent and substantially worse late in the rollout.

## Scientific interpretation

The current evidence supports a sharper diagnosis than "average reward is insufficient."

The information most consistently lost by the scalar heavy-policy return is:

> **how the preference-heavy policy differs from the matched center policy over time, especially whether the intended advantage persists into the late trajectory rather than decaying or reversing.**

The semantic endpoint is relational by design. The current training objective is largely heavy-policy absolute return. That mismatch appears more important than generic variance or maximum-excursion statistics.

A useful causal abstraction is:

    current objective
    = compress heavy trajectory to one scalar

    semantic competence
    = compare heavy vs center behavior
      + require directional consistency across matched suites
      + remain sensitive to where in the trajectory that advantage persists

The audit therefore points toward **relational persistence information** as the strongest missing information class.

## Decision

Reward redesign remains **NOT AUTHORIZED** by this audit alone.

Reasons:
1. the formal family-level comparator was contaminated by a tautological semantic-margin baseline;
2. the current sample count is 64 axis-transitions with only 9 PASS -> FAIL events;
3. strong temporal correlations identify an information class, but do not yet establish a causal reward term.

If further work is pursued, the next gate should be a narrowly predeclared validation of relational persistence descriptors on held-out policy transitions / seeds before introducing any reward term.

The preferred candidate family for that validation is:
- late heavy-vs-center advantage
- late-minus-early heavy-vs-center advantage
- final-16 heavy-vs-center advantage
- worst-16 heavy-vs-center advantage
- positive-advantage persistence

Generic heavy-only p90/p95/max excursion is lower priority based on this audit.

## Thesis wording

Recommended scope-bounded statement:

> Trajectory attribution indicated that semantic changes were substantially better aligned with the temporal persistence of matched heavy-versus-center behavioral advantages than with the heavy-policy objective return alone. Late-phase, final-window, and worst-window relational advantages tracked semantic-margin changes consistently across policy paths and objectives, and late relational advantage deteriorated in every observed PASS-to-FAIL transition. This suggests that the scalar training return discards relational temporal information that is relevant to semantic competence, although a causal reward redesign was not established.

Primary artifacts:
- `docs/contracts/diagnostics/trajectory-information-attribution-contract.md`
- `scripts/rl/trajectory_information_attribution_audit.py`
- `runs/trajectory_information_attribution_audit-2026-09-24/trajectory_information_attribution_report.json`
