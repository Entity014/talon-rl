# V2-B Objective-Interference Audit Verdict

Status: FROZEN — CROSS-OBJECTIVE OVERRIDE NOT PRIMARY; V2-C REMAINS OFF

Date: 2026-09-23

## Question

Does V2-B fail because heavy-objective semantic credit is overridden by competing T/A/O/S gradients after shared conditioning/FiLM?

## Read-only gradient audit

Frozen checkpoint: runs/v2b1_film_authority-2026-09-23/model_75.pt
Matched stochastic rollout seeds: 860001..860004.

At each heavy preference, objective-specific PPO actor gradients were measured separately and then compared with the actual weighted-combined gradient.

### Heavy-objective alignment

| Preference | combined cosine to heavy objective | other/heavy weighted norm ratio | heavy-vs-others cosine |
|---|---:|---:|---:|
| T-heavy | 0.875 | 0.519 | -0.714 |
| A-heavy | 0.991 | 0.201 | +0.654 |
| O-heavy | 0.995 | 0.133 | +0.591 |
| S-heavy | 0.970 | 0.353 | +0.549 |

O-heavy is the strongest counterexample to the override hypothesis: the combined update is almost collinear with the Orientation-specific gradient, and non-O weighted components are only ~13% of the O-heavy component norm.

The same result persists across parameter/representation locations for O-heavy:
- shared actor body: cosine to O = 0.994, override ratio = 0.138
- direct preference-input weights: cosine to O = 0.994, override ratio = 0.141
- FiLM parameters: cosine to O = 0.993, override ratio = 0.136
- actor head: cosine to O = 0.995, override ratio = 0.125
- post-FiLM hidden feature: cosine to O = 0.995, override ratio = 0.125
- gamma representation: cosine to O = 0.996, override ratio = 0.123
- pre-tanh mean: cosine to O = 0.995, override ratio = 0.125

Thus the O endpoint failure is not explained by other objectives numerically dominating the O-heavy update.

## Objective-vs-physical semantic credit control

A second read-only audit compared three score-function directions on the same frozen rollout:
1. PPO/GAE objective gradient
2. Monte-Carlo objective-return gradient
3. critic-independent physical-proxy return gradient

All-actor cosine results:

| Axis | GAE vs MC objective | GAE vs physical proxy | MC objective vs physical proxy |
|---|---:|---:|---:|
| T | 0.401 | 0.344 | -0.293* |
| A | 0.548 | 0.539 | 0.989 |
| O | 0.461 | 0.409 | 0.969 |
| S | 0.520 | 0.520 | 1.000 |

*The simple Tracking proxy used here is not equivalent to the normalized exponential Tracking objective and is not used for causal interpretation.

For A/O/S, Monte-Carlo objective gradients closely match their physical semantic proxies. This supports the objective definitions themselves as semantically aligned.

PPO/GAE directions are substantially rotated relative to MC/proxy directions, but this rotation is not unique to failing axes: Angular passes the endpoint gate while showing only ~0.54 GAE-to-proxy cosine. Therefore GAE rotation is real but is not sufficient by itself to explain O/S semantic failure.

## Causal status

- Conditioning authority insufficiency: REJECTED as primary unknown (B1 closed this).
- Foundation/critic collapse: REJECTED.
- Cross-objective weighted-gradient override: NOT SUPPORTED as primary cause.
- Orientation objective semantic sign error: NOT SUPPORTED; MC Orientation objective and physical tilt proxy align strongly.
- Smoothness objective semantic sign error: NOT SUPPORTED; MC Smoothness objective and action-rate proxy align essentially exactly.
- PPO/GAE vs long-horizon MC credit rotation: SUPPORTED, but not uniquely predictive of endpoint failure.
- Remaining locus: LOCAL CREDIT / TRAJECTORY REALIZATION / DYNAMICS COUPLING remains unresolved.

## Decision

V2-C is NOT authorized from this audit.

The next experiment should remain read-only and determine why an O-heavy update that is numerically dominated by the correct O objective does not realize an O-semantic endpoint in closed-loop behavior. The next gate should examine state/time-local credit and trajectory evolution under O-heavy versus matched center/other preferences, rather than adding conditioning capacity.