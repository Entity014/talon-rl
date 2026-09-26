# V2-B Lambda-Only Training Pilot Verdict

Status: FROZEN — LAMBDA=1.0 IMPROVES CREDIT FIDELITY BUT DOES NOT IMPROVE NET SEMANTIC CONTROL

Date: 2026-09-23

## Experimental contract

Paired training comparison:
- architecture: V2-B frozen
- Foundation V2 frozen
- initialization: identical V2-B0 checkpoint
- seed: 73001
- updates: 75
- critic support / refresh: identical
- PPO/action semantics: identical
- objectives/reward/normalization: identical
- semantic evaluator / thresholds / reset suites: identical

Only treatment:
- control: GAE lambda = 0.95
- treatment: GAE lambda = 1.00

Both training arms passed Foundation V2 guardrails.

## Foundation status at update 75

lambda=0.95:
- early critic EV: 0.5355
- late critic EV: 0.2807
- combined negative fraction: 0.05
- last-10 termination fraction: 0.0
- PPO max ratio error: 1.53e-5

lambda=1.00:
- early critic EV: 0.4165
- late critic EV: 0.2138
- combined negative fraction: 0.075
- last-10 termination fraction: 0.0
- PPO max ratio error: 3.05e-5

Thus both semantic evaluations are valid under the frozen foundation contract.

## Semantic comparison

| Metric | lambda=0.95 | lambda=1.00 |
|---|---:|---:|
| Tracking endpoint objective / physical | 0.75 / 0.50 | 0.25 / 0.50 |
| Angular endpoint objective / physical | **1.00 / 0.75 (PASS)** | 0.00 / 0.00 |
| Orientation endpoint objective / physical | 0.00 / 0.00 | **0.50 / 0.50** |
| Smoothness endpoint objective / physical | 0.75 / 0.25 | **0.75 / 0.75 (PASS)** |
| endpoint pass count | 1/4 (A) | 1/4 (S) |
| continuum monotonicity | **0.6458** | 0.6354 |
| endpoint-between fraction | **0.5208** | 0.4167 |
| survival | 1.0 | 1.0 |
| max tracking collateral ratio | 1.028 | **0.997** |
| critic H32 EV mean | **0.3335** | 0.3094 |
| critic negative fraction | **0.0875** | 0.1125 |

## Interpretation

The read-only lambda audit correctly established that lambda=1.0 tracks MC32 geometry more closely than lambda=0.95 across Orientation, Angular, and Smoothness.

However, that estimator-level gain is not sufficient to improve system-level semantic controllability.

Behaviorally, lambda=1.0 redistributes semantic success rather than accumulating it:
- Orientation improves materially from 0.00/0.00 to 0.50/0.50, but still fails.
- Smoothness becomes the only endpoint that passes.
- Angular, which passed under lambda=0.95, collapses to 0.00/0.00.
- continuum monotonicity decreases slightly.
- endpoint-between fraction decreases substantially.

This is not a foundation collapse: survival, critic guardrails, and PPO invariants remain valid.

## Causal conclusion

Supported:
- horizon-dependent credit geometry is real.
- lambda=1.0 gives a closer MC32 estimator than lambda=0.95.
- lambda=1.0 changes learned semantic allocation in behavior.

Rejected:
- better MC32 alignment is sufficient to improve overall 4D semantic control.
- lambda=1.0 is a justified replacement for lambda=0.95 based on current system-level evidence.

Best interpretation:
> Long-horizon credit fidelity is a real mechanism, but the remaining multi-objective semantic problem is not solved by a global scalar increase in GAE lambda. The intervention shifts which objectives succeed rather than producing consistent multi-axis gains.

## Decision

- Keep architecture escalation OFF.
- Do not adopt lambda=1.0 as the final training rule from this evidence.
- Do not tune lambda further as a scalar sweep without a new causal question.
- Current lambda=0.95 remains the conservative baseline because it preserves better overall continuum behavior and Angular semantics under the frozen contract.
- Any next experiment must address state/objective-dependent temporal allocation rather than global horizon length alone.