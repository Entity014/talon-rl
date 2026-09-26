# Semantic-Gate Margin / Robustness Audit Verdict

Status: **FROZEN — MIXED / INCONCLUSIVE FORMALLY; SUBSTANTIAL REAL COLLAPSE WITH MATERIAL BOUNDARY-SENSITIVE SUBSET**

Date: 2026-09-24

## Question

Are held-out semantic PASS -> FAIL events mostly genuine competence collapses, or are they largely artifacts of the discrete semantic PASS threshold?

## Exact continuous gate margin

For each channel, the frozen semantic gate requires at least 3/4 suites to have positive heavy-versus-center margin.

After discovery-only normalization, the exact continuous distance to that boundary is:

    G_obj  = second-smallest normalized objective suite margin
    G_phys = second-smallest normalized physical suite margin
    G_sem  = min(G_obj, G_phys)

Across all checkpoint-axis states:

    G_sem > 0  <=> frozen PASS

This identity was verified with **0 mismatches**.

## Held-out PASS -> FAIL events

There are 14 events.

Aggregate continuous-margin behavior:

    median source G_sem = +0.459
    median target G_sem = -2.474
    median Delta G      = -2.931

Thus the typical forgetting event crosses substantially beyond the decision boundary rather than merely changing sign by an infinitesimal amount.

## Small threshold sweep

Diagnostic suite-sign thresholds:

    tau in {-0.25, 0.00, +0.25}

The original PASS -> FAIL verdict survives at all three thresholds in:

    9 / 14 = 64.3%

Therefore nearly two-thirds of forgetting events are robust to a moderate normalized threshold perturbation.

Conversely, 5/14 events change classification under the threshold sweep, showing that evaluator sensitivity is not negligible.

## Distance-to-boundary

Boundary clearance:

    C = min(G_source, -G_target)

Results:

    C >= 0.25 : 9 / 14 = 64.3%
    C <  0.10 : 4 / 14 = 28.6%

Thus a clear majority cross the semantic boundary with appreciable normalized clearance, while roughly one quarter are genuinely near-boundary events.

## Exact paired reset bootstrap

All 4^4 = 256 paired reset resamples were enumerated exactly.

Results:

    bootstrap PASS->FAIL probability >=0.50 : 13 / 14
    bootstrap PASS->FAIL probability >=0.6875 : 10 / 14
    median bootstrap PASS->FAIL probability   : 0.6875

This indicates that most events remain more likely than not to appear as forgetting under reset resampling.

## Important contract limitation

The predeclared class-B rule required:

    bootstrap PASS->FAIL probability >=0.75

However, for a source checkpoint that passes exactly 3 of 4 suites, the maximum possible exact bootstrap probability of resampling a PASS source is:

    189 / 256 = 0.73828125

Therefore the 0.75 cutoff is combinatorially unreachable for the common 3-of-4 source-PASS case.

As a result:

    class B count = 0

must **not** be interpreted as evidence that no genuine collapses exist.

The formal predeclared aggregate verdict is therefore retained as:

> **MIXED / INCONCLUSIVE**

and the class-B count is treated as non-informative because of this design flaw.

No threshold is changed post hoc.

## Useful event classification despite the formal limitation

Under the frozen A/B/MIXED implementation:

    A — boundary/evaluator sensitive : 6 / 14 = 42.9%
    B — robust genuine collapse      : 0 / 14 (invalidated by impossible bootstrap cutoff)
    MIXED                            : 8 / 14 = 57.1%

The more reliable continuous diagnostics show:

- 64.3% survive the full threshold sweep;
- 64.3% have clearance >=0.25;
- 92.9% have bootstrap flip probability >=0.50.

Taken together, the evidence supports a mixed interpretation:

> evaluator discretization contributes materially to a minority/subset of events, but most observed semantic forgetting cannot be dismissed as threshold noise alone.

## Limiting semantic channel

At the target FAIL checkpoints:

    physical channel limiting : 8 / 14
    objective channel limiting: 6 / 14

There is no single-channel explanation for forgetting.

## Two residual counterexamples

### Seed 980001, u3 -> u4, Smoothness

This event is genuinely boundary-sensitive:

    G_source = +0.478
    G_target = -0.045
    clearance = 0.045

At tau=-0.25, the target returns to PASS.

Thus this previously puzzling residual failure is partly explained by evaluator boundary sensitivity.

### Seed 981001, u4 -> u5, Tracking

This event is substantially more robust:

    G_source = +0.253
    G_target = -0.592
    clearance = 0.253

The PASS -> FAIL flip survives all three threshold settings.

Therefore the two residual counterexamples are not equivalent:

- Smoothness is near-boundary / evaluator-sensitive;
- Tracking represents a more substantial semantic loss that remains unexplained by mean relation or preference ordering.

## Scientific interpretation

The correct updated conclusion is not:

> semantic forgetting is merely a threshold artifact.

Nor is it:

> every PASS -> FAIL is a large, robust competence collapse.

Instead:

> semantic forgetting is heterogeneous. Most held-out events exhibit substantial continuous margin loss and remain stable under threshold/reset perturbations, while a meaningful subset lies close enough to the evaluator boundary that the discrete PASS label amplifies relatively small behavioral changes.

This distinction matters for thesis wording and future evaluation design.

## Decision

No training method is authorized.

No semantic threshold or PASS rule is changed retrospectively.

For the thesis:
- report PASS/FAIL for compatibility with the frozen evaluation contract;
- accompany it with continuous semantic gate margin where possible;
- distinguish robust competence collapse from boundary-sensitive flips in discussion.

Future work may use confidence-aware or continuous-margin reporting, but this audit does not authorize changing the current thesis evaluation results.

## Thesis wording

> A robustness audit showed that semantic PASS-to-FAIL transitions were heterogeneous rather than uniformly threshold-driven. The median continuous gate margin moved from +0.459 at the source checkpoint to -2.474 at the target, and 64.3% of forgetting events remained PASS-to-FAIL under normalized threshold perturbations of ±0.25. However, 28.6% of events had boundary clearance below 0.10, demonstrating a meaningful evaluator-sensitive subset. Thus semantic forgetting is largely real but the discrete PASS gate can amplify near-boundary changes in some cases; continuous semantic margins should therefore accompany binary competence labels.

Primary artifacts:
- `docs/contracts/diagnostics/semantic-gate-robustness-audit-contract.md`
- `scripts/rl/semantic_gate_robustness_audit.py`
- `runs/semantic_gate_robustness_audit-2026-09-24/semantic_gate_robustness_report.json`
