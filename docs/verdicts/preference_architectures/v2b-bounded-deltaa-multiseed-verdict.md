# V2-B Bounded Delta-a Multi-Seed Validation Verdict

Status: FROZEN — OPTIMIZATION-SAFETY CONTRACT REPRODUCES, BUT RETENTION EFFICACY DOES NOT CONFIRM ACROSS SEEDS.

Date: 2026-09-24

## Candidate

Preference-conditioned action-response rehearsal:

Delta a_i(s) = pi(s,w_i) - pi(s,w_center)

with the previously validated bounded rehearsal budget:

alpha_t = min(beta_cap, rho * ||g_mixed|| / (||g_rehearsal|| + eps))

rho = 0.25.

No semantic-outcome estimator or surrogate is used.

## Validation contract

Three independent training seeds:
- 980001
- 981001
- 982001

Each seed uses paired control and bounded-Delta-a arms with:
- identical initialization,
- identical architecture / critic / lambda,
- identical mixed MORL objective,
- identical nominal finite update norm,
- identical fixed endpoint evaluation suites,
- identical semantic reference distribution and activation rule.

Only rehearsal OFF versus bounded Delta-a rehearsal ON changes.

## Paired result

PASS-event difference, bounded Delta-a minus control:
- seed 980001: +2
- seed 981001: -2
- seed 982001: -1

Total PASS events:
- control: 20
- bounded Delta-a: 19

Mean paired PASS-event effect:
- -0.333
- sample SD: 2.082

Mean semantic-score difference:
- seed 980001: +0.00391
- seed 981001: -0.00781
- seed 982001: -0.01953

Across seeds:
- mean effect = -0.00781
- sample SD = 0.01172
- positive semantic-score effect in 1/3 seeds

Therefore the single-seed persistence gain does not reproduce as a robust method-level retention advantage.

## Optimization safety does reproduce

Across the three seeds:
- mean cosine between mixed direction and bounded total direction = 0.9745
- minimum observed mixed-total cosine = 0.9698
- mean preference-separation difference versus control = +0.00013, effectively unchanged
- active rehearsal updates are normally capped at weighted rehearsal / mixed-gradient ratio = 0.25

For seed 982001 one first nonzero rehearsal update used ratio 0.1968 because beta_cap became the tighter bound; subsequent active updates reached approximately 0.25. This is valid behavior under the predeclared min(beta_cap, rho-budget) rule and is not a contract violation.

Thus the multi-seed result reproduces:
- no rehearsal-gradient hijack,
- no preference-authority collapse,
- no parameter-step collapse,
- bounded rehearsal contribution.

## Axis-level retention

The effects are heterogeneous rather than uniformly beneficial.

Examples:
- S retention improves in seeds 980001 and 982001 but worsens in seed 981001.
- A retention is worse in seed 980001, modestly better in seed 981001, and effectively tied in retained-pass fraction in seed 982001.
- O worst-case forgetting improves in seeds 981001 and 982001, but this does not translate into a reliable total PASS-event or semantic-score gain.
- T worst-case forgetting improves in seeds 981001 and 982001, while final T scores can still be lower under rehearsal.

No axis provides a clean, monotonic cross-seed retention advantage sufficient to rescue the overall method claim.

## Interpretation

The bounded-gradient mechanism and the Delta-a content must now be separated in status.

Supported:
- bounded rehearsal is a stable optimization mechanism;
- rho = 0.25 prevents replay-gradient dominance;
- Delta-a rehearsal is technically well-behaved and remains a useful controlled retention baseline.

Not supported:
- bounded Delta-a rehearsal produces a robust semantic-retention improvement over no rehearsal.

The earlier single-seed 6-vs-4 PASS-event result was a genuine positive instance, but it is not representative enough to support a method-level efficacy claim.

## Causal status

- rehearsal-budget control: VALIDATED
- score-function semantic-outcome branch: CLOSED
- pathwise semantic-outcome branch: CLOSED
- local surrogate semantic-outcome branch: CLOSED
- bounded Delta-a optimization safety: VALIDATED
- bounded Delta-a retention efficacy across seeds: NOT CONFIRMED / REJECTED AS ROBUST CLAIM
- bounded Delta-a as controlled rehearsal baseline: RETAIN
- bounded Delta-a as promoted final retention method: DO NOT PROMOTE

## Decision

For the thesis, present bounded Delta-a rehearsal as a carefully validated retention-control mechanism and negative/limited method result, not as a robustly superior retention algorithm.

The strongest evidence-supported conclusion is:

> Bounding rehearsal-gradient contribution solves the optimization-stability problem, but preserving preference-conditioned action-response differences is not sufficient to deliver consistent semantic retention across training seeds.

Do not reopen the semantic-outcome branch within the current formulation.

A new retention method would require a new representation/content hypothesis rather than additional coefficient or estimator tuning.
