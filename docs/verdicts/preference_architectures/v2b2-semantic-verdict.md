# V2-B2 Semantic Verdict

Status: FROZEN — V2-B2 FAIL; SINGLE-SITE FiLM SEMANTICALLY INSUFFICIENT

Date: 2026-09-23

## Frozen semantic contract result

V2-B was evaluated with the exact Foundation V2 matched-reset semantic contract and unchanged thresholds.

Endpoint results:
- Tracking: FAIL — objective correctness 0.75, physical correctness 0.50
- Angular: PASS — objective correctness 1.00, physical correctness 0.75
- Orientation: FAIL — objective correctness 0.00, physical correctness 0.00
- Smoothness: FAIL — objective correctness 0.75, physical correctness 0.25

Continuum:
- monotonicity fraction: 0.645833 < 0.65
- endpoint-between fraction: 0.520833 < 0.65

Guardrails:
- survival: 1.0
- max tracking ratio to center: 1.0281
- critic H32 EV mean: 0.3335
- critic negative fraction: 0.0875
- critic mean absolute bias: 0.0344

Therefore the failure is semantic, not a foundation/survival/critic collapse.

## Foundation-V2 architecture progression

| Architecture | Pairwise action distance | preference Jacobian | endpoint passes | continuum monotonicity | endpoint-between | survival |
|---|---:|---:|---:|---:|---:|---:|
| RV1 direct | 0.04777 | 0.02988 | 0/4 | 0.60938 | 0.40278 | 1.0 |
| V2-A + embedding | 0.06215 | 0.03249 | 1/4 (T) | 0.62500 | 0.47917 | 1.0 |
| V2-B + one-site FiLM | 0.06248 | 0.02910 | 1/4 (A) | 0.64583 | 0.52083 | 1.0 |

Interpretation:
- Conditioning expressivity increased action separation and improved continuum metrics progressively.
- The improvement did not become globally correct 4D semantic control.
- Endpoint success moved from Tracking in V2-A to Angular in V2-B rather than accumulating across axes.
- Orientation became the clearest failure in V2-B (0/4 objective and 0/4 physical correctness).
- V2-B1 proved FiLM has causal authority, so V2-B2 failure is not explained by an inactive modulation branch.

## Decision

V2-B2 FAIL.

Single-site FiLM adds real preference-conditioned authority and improves interpolation metrics, but is insufficient for the frozen continuous 4D semantic contract.

V2-B3 multi-seed validation is not authorized.

V2-C is not automatically authorized by this result. Before adding more modulation sites or mechanisms, the next experiment must establish whether the remaining failure is an axis-specific credit/interference problem (especially Orientation/Smoothness) or merely insufficient conditioning capacity.

No adapters, routing, multi-site FiLM, or auxiliary semantic losses are authorized yet.