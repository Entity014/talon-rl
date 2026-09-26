# V2-B Semantic-Outcome Rehearsal Gate Interim Verdict

Status: FROZEN — SEMANTIC-OUTCOME TARGET DOES NOT OUTPERFORM BOUNDED ACTION-RESPONSE REHEARSAL UNDER THE CURRENT SCORE-FUNCTION ESTIMATOR. CONTENT-LEVEL REJECTION IS BLOCKED BY ESTIMATOR-QUALITY CONFOUND.

Date: 2026-09-24

## Controlled three-arm comparison

All arms use the same V2-B foundation, nominal step norm, reset/evaluation suites, rho=0.25 bounded rehearsal budget, beta cap, and per-arm PASS-triggered memory activation.

Arms:
- Control: mixed MORL only.
- A: bounded-budget action-response rehearsal, retaining Delta a_i(s)=pi(s,w_i)-pi(s,w_center).
- B: bounded-budget semantic-outcome rehearsal using matched heavy-vs-center multi-step rollout margins and a score-function recovery gradient.

Outcome rehearsal uses only positive reference semantic evidence and a kappa=0.5 soft hinge. It does not backpropagate through simulator physics and does not use the critic as the rehearsal target.

## Sanity checks

At u1 the semantic-outcome arm has zero active deficient windows, zero rehearsal budget, and matches control/A exactly.

After rehearsal activates, both A and B respect the same bounded budget:
- weighted rehearsal / mixed gradient = 0.25,
- mixed-total cosine approximately 0.97,
- full nominal step retained.

Thus gradient-budget and step-size confounds remain closed.

## Endpoint availability

Total PASS events over 8 updates:
- Control: 4
- Action-response rehearsal: 6
- Semantic-outcome rehearsal: 3

Activation:
- Action-response: O@u2, A@u5, S baseline.
- Semantic-outcome: O@u2, T@u3, S baseline.

## Forgetting

Max forgetting:

| axis | control | action-response | semantic-outcome |
|---|---:|---:|---:|
| T | 0.500 | 0.500 | 0.625 |
| A | 0.500 | 0.625 | 0.375 |
| O | 0.500 | 0.500 | 0.625 |
| S | 0.750 | 0.750 | 0.750 |

Semantic-outcome rehearsal changes which competence is acquired but does not improve overall retention.

Retained-PASS fraction after first PASS:
- Semantic T: 0.0
- Semantic O: 0.0
- Semantic S: 0.125

## Important limitation

The action-response rehearsal gradient is deterministic and directly differentiable through the actor on stored states.

The semantic-outcome rehearsal gradient is instead estimated by a likelihood-ratio / score-function surrogate on stochastic matched rollouts.

Therefore the current result does not isolate memory content alone: semantic content and rehearsal-gradient estimator differ simultaneously.

## Decision

Do not promote the current semantic-outcome rehearsal implementation.

Do not yet conclude that semantic-outcome content is intrinsically worse than action-response content.

Next gate: fixed-policy, fixed-reference read-only audit of semantic-outcome rehearsal-gradient reproducibility/SNR across repeated common-reset rollouts and action-noise replicas.

If the semantic-outcome gradient is stable and reproducible but retention remains worse, the content/mapping hypothesis is weakened strongly.

If the semantic-outcome gradient has poor SNR or unstable direction, repair or average the estimator before making a content-level judgment.