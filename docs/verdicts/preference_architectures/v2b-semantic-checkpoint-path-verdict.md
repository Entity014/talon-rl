# V2-B Lambda Semantic Checkpoint-Path Verdict

Status: FROZEN — ANGULAR COLLAPSE IS A LATE TRAINING-PATH EVENT, NOT A STATIC CREDIT-DIRECTION FAILURE

Date: 2026-09-24

## Design

Exact frozen semantic evaluator applied at u10/u25/u50/u75 for both paired training arms:
- V2-B lambda=0.95 control
- V2-B lambda=1.0 treatment

Same endpoint/continuum reset suites, thresholds, survival/collateral/critic guardrails as V2-B2.

## Semantic path

### lambda=0.95

| update | endpoint passes | T obj/phys | A obj/phys | O obj/phys | S obj/phys | monotonicity | endpoint-between |
|---:|---|---|---|---|---|---:|---:|
| 10 | O | .75/.50 | .50/.75 | 1.00/.75 | .25/.00 | .594 | .292 |
| 25 | none | .25/.25 | .25/.25 | .25/.25 | .50/.50 | .599 | .403 |
| 50 | O | .25/.25 | .75/.50 | 1.00/1.00 | .50/.50 | .573 | .347 |
| 75 | A | .75/.50 | 1.00/.75 | .00/.00 | .75/.25 | .646 | .521 |

### lambda=1.0

| update | endpoint passes | T obj/phys | A obj/phys | O obj/phys | S obj/phys | monotonicity | endpoint-between |
|---:|---|---|---|---|---|---:|---:|
| 10 | O,S | .00/.25 | .50/.50 | .75/.75 | .75/.75 | .604 | .417 |
| 25 | none | .50/.25 | .50/.75 | .50/.50 | .50/.75 | .568 | .389 |
| 50 | T,A | .75/.75 | .75/.75 | .50/.50 | .25/.25 | .615 | .340 |
| 75 | S | .25/.50 | .00/.00 | .50/.50 | .75/.75 | .635 | .417 |

## Main finding

Angular behavior under lambda=1.0 is not monotonically poor:
- u10: partial A semantics (.50/.50)
- u25: .50/.75
- u50: A endpoint PASSES (.75/.75)
- u75: collapses to .00/.00

Therefore final Angular failure is a late training-path event between u50 and u75.

## Relation to gradient geometry

At the same late checkpoints, fresh on-policy gradient audits do not show loss of Angular credit authority:

Mixed training-style batch, all-actor:
- u50 control A combined cosine ~.661; treatment ~.709
- u75 control ~.537; treatment ~.589
- u75 A contribution share control ~.258; treatment ~.255

A-heavy dedicated batches are even more strongly aligned:
- u75 control cos(combined,A) ~.991
- u75 treatment ~.993

Thus Angular semantic collapse occurs while instantaneous Angular gradient alignment remains healthy.

## Interpretation

The evidence rejects a simple model in which lambda=1 directly rotates or suppresses Angular gradients until Angular behavior fails.

Instead, semantic winners are non-stationary over training:
- lambda=1: early O/S -> mid T/A -> late S
- lambda=.95: early O -> mid O -> late A

This indicates accumulated shared-policy optimization-path dependence. Repeated updates move the policy through regions where the same locally valid objective gradients realize different closed-loop semantics.

Lambda=1 also produces substantially larger mixed-batch gradient norms (~1.6-2.1x control across audited checkpoints), making effective step scale / optimizer trajectory a plausible contributor even though per-step direction remains valid.

## Causal status

- Static per-objective credit fidelity failure: rejected.
- Static A-heavy suppression: rejected.
- Static mixed-batch Angular suppression: rejected.
- Final Angular failure as an always-present lambda=1 property: rejected.
- Late u50->u75 semantic collapse: established.
- Non-stationary shared-policy optimization path: strongly supported.
- Effective gradient/step-scale amplification: supported contributor.
- Visitation/dynamics feedback: still potentially involved.

## Decision

Do not add architecture, vector lambda, adaptive lambda, or gradient surgery yet.

The next clean diagnostic should focus specifically on the lambda=1 u50->u75 transition. A high-value test is a parameter-path / update-scale audit: compare blockwise parameter displacement and semantic behavior along the u50->u75 path, and test whether reducing only actor step scale (while retaining lambda=1 credit geometry) prevents the late Angular collapse.