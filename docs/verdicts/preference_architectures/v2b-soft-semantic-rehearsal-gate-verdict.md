# V2-B Soft Semantic Reference-Distribution Rehearsal Gate Verdict

Status: FROZEN — FIXED-WEIGHT SOFT REHEARSAL DOES NOT IMPROVE RETENTION; REHEARSAL GRADIENT BUDGET GROWS AND DOMINATES THE MIXED UPDATE.

Date: 2026-09-24

## Question

Can a soft rehearsal loss over a broader semantic reference distribution improve retention without the plasticity collapse caused by hard constraints?

## Candidate

Treatment uses the same mixed MORL update as control plus a soft semantic rehearsal term.

Rehearsal target is not exact action cloning. For each acquired semantic axis i, it preserves the preference-conditioned action response relative to center:

`Delta a_i(s) = pi(s, w_i) - pi(s, w_center)`

Reference distribution per acquired axis:
- 4 matched reset seeds,
- 4 trajectory phases: 0, 8, 16, 24,
- heavy-preference and center visitation,
- approximately 256 states per axis.

Rehearsal loss is normalized by the reference semantic action-separation scale.

## Coefficient calibration

A naive beta=0.25 pilot was stopped because the rehearsal gradient was effectively negligible.

The final v2 candidate therefore uses a one-time, outcome-independent gradient-scale calibration:
- u1: control and treatment are identical mixed updates,
- before u2: choose one fixed beta so weighted rehearsal gradient norm is 25% of mixed gradient norm,
- freeze that beta for u2-u8.

Calibration result:
- mixed gradient norm = 6.5137,
- rehearsal gradient norm = 0.6521,
- beta = 2.4970,
- weighted rehearsal / mixed gradient ratio at calibration = 0.25.

No projection, backtracking, trust-region feasibility check, or adaptive beta is used after calibration.
Every update uses the full nominal diagnostic step norm.

## Plasticity magnitude

Unlike the hard-retention candidates, parameter-step magnitude does not collapse.

The treatment remains free to take the same nominal finite step at every update and acquires new semantic axes:
- O first PASS at u2,
- T first PASS at u3.

Preference-conditioned action separation also remains near the control level (~0.060-0.062).

Thus the method does preserve basic optimization movement and preference authority.

## Retention outcome

Total endpoint-PASS events over 8 updates:
- control: 4
- soft rehearsal: 3

Max forgetting:

| axis | control | rehearsal |
|---|---:|---:|
| T | 0.50 | 0.50 |
| A | 0.50 | 0.625 |
| O | 0.50 | 0.50 |
| S | 0.75 | 0.75 |

Retained-PASS fraction after first PASS:
- control O: 0.20; rehearsal O: 0.167
- control S: 0.125; rehearsal S: 0.00
- rehearsal T is acquired at u3 but retained fraction afterward is 0.00
- treatment never acquires A.

Therefore the fixed-weight soft rehearsal candidate does not improve semantic retention under the primary contract.

## Why: rehearsal gradient budget grows after calibration

The one-time beta remains fixed, but the rehearsal gradient magnitude grows as the policy moves and the retained reference set expands.

Weighted rehearsal / mixed gradient norm ratio:
- u1: 0.00
- u2: 0.25
- u3: 0.797
- u4: 2.119
- u5: 5.374
- u6: 3.904
- u7: 4.537
- u8: 3.802

Mean ratio over the path: ~2.60.

Corresponding cosine between the raw mixed direction and total mixed+rehearsal direction:
- u2: 0.970
- u3: 0.778
- u4: 0.426
- u5: 0.176
- u6: 0.254
- u7: 0.207
- u8: 0.259

Mean cosine over the full path: ~0.51.

Thus fixed beta avoids shrinking the parameter step but does not preserve plasticity in direction space. The rehearsal term progressively hijacks the update geometry.

## Causal interpretation

The result rejects a simple fixed-weight formulation:

> A fixed beta calibrated once is sufficient to maintain a stable retention-plasticity balance as semantic references accumulate.

That statement is false in this experiment.

The broader rehearsal direction is not disproven by this result because:
- treatment can acquire O and T while retaining full step magnitude,
- the failure coincides with a large, directly measured growth of rehearsal-gradient dominance,
- the issue is the rehearsal gradient budget, not hard feasibility collapse.

## Causal status

- Hard retention family: CLOSED as insufficient/plasticity-limiting.
- Soft distribution-level rehearsal: METHOD DIRECTION STILL PLAUSIBLE.
- Fixed beta soft rehearsal: REJECTED.
- Full nominal step magnitude under soft rehearsal: PRESERVED.
- New-axis acquisition under soft rehearsal: ESTABLISHED.
- Retention improvement under fixed beta: REJECTED.
- Unbounded rehearsal-gradient accumulation: ESTABLISHED.

## Decision

Do not promote this fixed-beta rehearsal rule.

The next justified refinement is not a larger rehearsal set or a larger beta. It is an explicitly bounded rehearsal-gradient budget.

A clean next candidate would retain the same reference distribution and semantic response target but cap the rehearsal contribution to a predeclared fraction of the current mixed-gradient norm, e.g. 25%, at every update:

`g_total = g_mixed + alpha_t g_rehearsal`

with `alpha_t` chosen only to enforce a fixed gradient-budget ratio, not from semantic endpoint outcomes.

This tests the same distribution-level retention hypothesis while preventing replay accumulation from replacing the current learning objective.