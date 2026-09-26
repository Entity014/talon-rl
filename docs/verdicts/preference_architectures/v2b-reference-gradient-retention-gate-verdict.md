# V2-B Reference-Gradient Retention Gate Verdict

Status: FROZEN — PARTIAL RETENTION SIGNAL, BUT FIRST-ORDER GRADIENT CONSTRAINT IS NOT SUFFICIENT FOR BEHAVIOR-LEVEL SEMANTIC RETENTION

Date: 2026-09-24

## Question

Can a temporal reference-gradient constraint preserve semantic competence that has already been acquired, while allowing continued mixed-objective learning?

## Design

Same theta0 as the repeated-update audit: lambda=1 u75 V2-B checkpoint.

Control:
- 8 repeated mixed scalarized updates.

Treatment:
- same mixed updates, same fresh on-policy batches, same fixed step norm (0.5x nominal late actor delta),
- dynamically retain any semantic axis that has ever reached PASS,
- recompute each active reference descent direction on the current policy using a fixed matched reset seed for that axis,
- project the proposed mixed descent direction into the intersection of first-order half-space constraints d dot r_i >= 0,
- renormalize to the exact same parameter step norm as control.

Baseline semantic state:
- T fail, score 0.375
- A fail, score 0.000
- O fail, score 0.500
- S PASS, score 0.750

Thus S is retained from checkpoint 0. Treatment additionally activated O at update 1 and A at update 6.

## Endpoint-PASS event count

Across the 8 post-update checkpoints:

Control PASS events:
- A: 1
- O: 2
- S: 1
- T: 0
- total: 4

Retention treatment PASS events:
- A: 2
- O: 3
- S: 3
- T: 0
- total: 8

The treatment therefore produces a real positive signal: semantic competence appears more often under the retention-constrained path.

## Pass-retention after competence acquisition

Retained PASS fraction after first PASS:

| axis | control | retention |
|---|---:|---:|
| S | 0.125 | **0.375** |
| O | 0.200 | **0.286** |
| A | 1.000* | 0.500 |

*Control first acquires A only at the final checkpoint u8, so the value 1.0 is vacuous: there are no later checkpoints on which retention can fail.

Treatment acquires A earlier at u6 and recovers A again at u8, demonstrating continued learning rather than a frozen policy.

## Worst-case forgetting is not consistently improved

Max semantic forgetting from historical best score:

| axis | control | retention |
|---|---:|---:|
| T | 0.500 | **0.375** |
| A | **0.500** | 0.750 |
| O | **0.500** | 0.625 |
| S | 0.750 | 0.750 |

So the intervention does not satisfy a strong retention criterion. A and O can forget more severely than under control.

## The decisive counterexample: first-order constraints can hold while semantic competence is lost

Update 7 treatment:
- active retained axes: A, O, S
- all post-projection reference directional constraints satisfied:
  - A dot applied direction ≈ +0.018
  - O dot applied direction ≈ +0.193
  - S dot applied direction ≈ 0.000
- raw-to-projected cosine ≈ 0.99993

Yet after that update:
- A score falls from 1.00 at u6 to 0.25, FAIL
- O score = 0.25, FAIL
- S score = 0.50, FAIL

Thus satisfying d dot r_i >= 0 for all retained objective reference gradients does not guarantee behavior-level semantic retention.

A similar failure occurs at u3: O and S reference constraints are satisfied after projection, but both O and S endpoints fail.

## Projection magnitude

The projection usually changes the mixed direction only modestly:
- raw-to-projected cosine ranges roughly 0.985 to 1.000 when active.

This confirms that instantaneous mixed gradients are often already close to the retained local-gradient half-spaces. The semantic failure therefore cannot be explained solely by obvious first-order gradient conflict.

## Conditioning and learning are not frozen

Preference-conditioned pairwise action separation remains near the control level (~0.060-0.062).
Cumulative parameter displacement remains comparable to control.
Treatment acquires new O and A PASS states during the sequence.

Therefore the positive/negative results are not caused by freezing the policy or removing preference authority.

## Causal interpretation

The intervention supports two conclusions simultaneously:

1. Temporal retention constraints have useful signal: they increase the frequency with which semantic endpoints are present and improve S/O pass retention modestly.

2. A first-order reference-gradient constraint is not the correct abstraction for the full problem. Behavior-level semantics can disappear even when every retained local-gradient half-space constraint is satisfied.

This is consistent with the prior diagnosis that local gradient correctness is necessary but not sufficient for semantic retention under finite nonlinear shared-policy updates.

## Causal status

- Temporal retention as a method direction: SUPPORTED.
- Reference-gradient projection has measurable benefit: SUPPORTED.
- First-order gradient projection sufficient for semantic retention: REJECTED.
- Stable reduction of max forgetting: NOT ESTABLISHED.
- Policy freezing as explanation: REJECTED.
- Preference-authority collapse: REJECTED.

## Decision

Do not promote reference-gradient projection alone to the final method candidate.

The next intervention should preserve a reference quantity closer to the actual semantic contract, not merely the local gradient. The cleanest next candidate is a reference-behavior / preference-conditioned action trust region on retained semantic reset states:

- when an axis first reaches PASS, snapshot reference actions (and optionally semantic physical proxies) on its matched reference states;
- during later mixed updates, constrain the change in those retained-axis actions / semantic outputs;
- allow unconstrained learning elsewhere;
- evaluate with the exact same retention trace and max-forgetting metrics.

This directly targets the failure demonstrated here: behavior can drift even while local reference gradients remain non-conflicting.