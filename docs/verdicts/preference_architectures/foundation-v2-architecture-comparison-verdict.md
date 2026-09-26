# Foundation V2 Architecture Comparison Verdict

Status: FROZEN — RV1 FAIL; V2-A FAIL; V2-A INSUFFICIENT; V2-B MAY BE AUTHORIZED

Date: 2026-09-23

## Foundation V2

Both architectures were retrained from function-preserving initializations with the same architecture-agnostic foundation:
- same seed: 73001
- same 75-update budget
- same validated 4D objectives
- same repaired PPO/action-logprob semantics
- same frozen critic body
- same objective-specific linear heads
- same ridge lambda = 1
- same 6 current-policy anchor specs
- same recent adaptive pool of 24 support candidates
- both early-H32 and late-H32 units retained
- head refresh before advantage computation every update
- additional current-policy head refresh before saved evaluation checkpoints
- same matched endpoint/continuum semantic evaluator and thresholds

## Foundation status at update 75

RV1:
- pairwise fixed-state action distance: 0.04777
- preference Jacobian norm: 0.02988
- early critic EV mean: 0.5182
- late critic EV mean: 0.3173
- late negative fraction: 0.000
- last-10 termination fraction: 0.000
- FOUNDATION-V2 INTERNAL PASS

V2-A:
- pairwise fixed-state action distance: 0.06215
- preference Jacobian norm: 0.03249
- early critic EV mean: 0.5056
- late critic EV mean: 0.2481
- late negative fraction: 0.100
- last-10 termination fraction: 0.000
- FOUNDATION-V2 INTERNAL PASS

## Semantic evaluation

RV1:
- T endpoint: FAIL (objective 0.50, physical 0.25)
- A endpoint: FAIL (objective 0.50, physical 0.25)
- O endpoint: FAIL (objective 0.75, physical 0.50)
- S endpoint: FAIL (objective 0.25, physical 0.25)
- continuum monotonicity: 0.6094 < 0.65
- endpoint-between fraction: 0.4028 < 0.65
- critic EV mean: 0.2419; negative fraction: 0.1375
- survival: 1.0
- semantic verdict: FAIL

V2-A:
- T endpoint: PASS (objective 1.00, physical 1.00)
- A endpoint: FAIL (objective 0.75, physical 0.50)
- O endpoint: FAIL (objective 0.50, physical 0.75)
- S endpoint: FAIL (objective 0.25, physical 0.25)
- continuum monotonicity: 0.6250 < 0.65
- endpoint-between fraction: 0.4792 < 0.65
- critic EV mean: 0.2812; negative fraction: 0.1625
- survival: 1.0
- semantic verdict: FAIL

## Interpretation

Foundation amendment does not rescue RV1 direct conditioning: RV1 remains semantically insufficient on a clean, architecture-robust critic foundation.

V2-A learned embedding produces stronger preference-conditioned action separation and partial semantic improvement (notably Tracking endpoint, modest continuum gains), but it does not satisfy the predeclared continuous 4D semantic contract.

Therefore the evidence supports:
> Minimal learned preference embedding is not sufficient to solve the semantic-conditioning problem under the validated Foundation V2 contract.

This does not imply that embedding is harmful or useless; it produced partial gains. It establishes only that this minimal escalation is insufficient.

## Decision matrix result

Predeclared matrix outcome: RV1 FAIL + V2-A FAIL.

Decision:
- RV1 remains rejected as final thesis method under the tested direct-conditioning architecture.
- V2-A is insufficient as the final method.
- V2-B may be authorized as the next minimal escalation.
- Any V2-B experiment must keep Foundation V2, rewards, PPO, critic, support, seeds, semantic evaluator, and thresholds frozen.
- The next treatment should modify only actor conditioning, e.g. one minimal FiLM/modulation site, not adapters/routing/multiple mechanisms simultaneously.