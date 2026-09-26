# V2-B Short-Horizon Trajectory Semantic Retention Gate Verdict

Status: FROZEN — EARLY STOPPED. SHORT-HORIZON TRAJECTORY RETENTION IS NOT SUFFICIENT AND REMAINS PLASTICITY-LIMITING.

Date: 2026-09-24

## Question

Can semantic competence be preserved more faithfully by constraining short matched rollout outcomes rather than local gradients or fixed state-action snapshots?

## Candidate

When axis i reaches PASS:
- freeze matched 32-step reference rollouts under heavy preference w_i and center preference,
- retain the heavy-vs-center objective margin for axis i,
- retain the corresponding physical semantic margin,
- require survival >= 0.95.

Later mixed updates are accepted only if retained axes preserve at least 50% of their positive reference objective and physical margins on the short rollout.

If the raw mixed direction violates retention:
- rotate toward current axis-specific recovery directions,
- test the actual finite candidate through the short rollout,
- backtrack step magnitude if necessary.

Control uses the same effective step magnitude as treatment at every update.

## Early results

u1:
- effective step ratio = 0.25 of nominal,
- treatment retains S PASS and acquires O PASS,
- magnitude-matched control retains only S.

u2:
- effective step ratio = 0.0625,
- treatment retains O and S PASS,
- control has no passing endpoint.

These results show a real positive semantic-retention signal.

## Decisive counterexample

At u3, both retained short-rollout constraints are satisfied:
- S short objective/physical/survival constraint: PASS,
- O short objective/physical/survival constraint: PASS.

Yet full 64-step endpoint evaluation gives:
- S: PASS,
- O: semantic score 0.125, FAIL.

At u4, short-rollout constraints again remain satisfied for both O and S, but full endpoints are:
- O: score 0.125, FAIL,
- S: score 0.50, FAIL.

Therefore:
> Preserving the retained semantic outcome over a 32-step matched rollout does not guarantee preservation of the full 64-step closed-loop semantic endpoint.

## Plasticity failure

The feasible update shrinks rapidly:
- u1: 0.25 x nominal,
- u2: 0.0625 x nominal,
- u3: 0.03125 x nominal,
- u4: 0.0078125 x nominal.

At u4 the applied direction is also almost orthogonal/slightly opposite to the raw mixed direction:
- cos(raw, applied) ≈ -0.034.

This means the candidate fails the second primary requirement as well: preserving short semantic rollouts does not maintain adequate optimization plasticity.

## Why the gate was stopped at u4

The two primary method goals were already violated simultaneously:
1. retention failure: retained short semantic constraints passed while full semantic endpoints failed;
2. plasticity failure: feasible step collapsed below 1% of nominal.

Continuing to u8 would add compute but would not change the method-selection conclusion for this exact candidate contract.

## Comparison across retention levels

Gradient-space retention:
- weakest positive signal,
- local constraints can hold while behavior is forgotten.

Static action-space retention:
- stronger signal and more PASS events,
- but fixed action anchors still fail to preserve trajectory semantics and eventually squeeze plasticity.

32-step trajectory-semantic retention:
- initially stronger semantic retention than magnitude-matched control,
- but still fails to preserve 64-step endpoint semantics,
- and becomes even more restrictive as retained constraints accumulate.

## Causal interpretation

The evidence now supports a hierarchy:

> Retention must be defined at a broader distributional/trajectory level than a single local gradient, a fixed action snapshot, or one short matched rollout.

The competence being retained is not a point property. It depends on a broader closed-loop state-distribution and long-horizon outcome structure.

## Causal status

- Retention direction: VALIDATED.
- Gradient retention: TOO WEAK.
- Static action retention: BENEFICIAL BUT INSUFFICIENT.
- 32-step trajectory semantic retention: BENEFICIAL EARLY BUT INSUFFICIENT.
- Full endpoint retention from short-rollout preservation: REJECTED.
- Plasticity preservation under accumulating trajectory constraints: REJECTED.
- Need for broader semantic reference distribution / rehearsal: SUPPORTED.

## Decision

Do not promote the short-horizon trajectory trust-region candidate.

The next justified method family is broader semantic rehearsal / reference-distribution retention rather than increasingly hard trust regions on a small retained set.

A cleaner next candidate would use:
- a compact replay/reference distribution spanning multiple matched resets and trajectory phases for each acquired semantic axis,
- soft semantic rehearsal losses or distillation over that distribution,
- no hard per-update feasibility requirement,
- explicit stability-plasticity measurement against an unconstrained magnitude-matched baseline.

This would test whether retaining a distribution of semantic states/trajectories preserves competence without collapsing the feasible update space.