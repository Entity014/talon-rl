# V2-C1 Modular Authority / Specialization Verdict

Status: **FROZEN — V2-C1 FAIL; V2-C2 NOT AUTHORIZED; V2-C BRANCH CLOSED**

Date: 2026-09-24

## Question

Does the V2-C modular path become genuinely preference-conditioned and specialized before any semantic endpoint judgment is attempted?

V2-C1 intentionally does **not** evaluate semantic endpoint success.

## Frozen contract

- V2-C0 exact function-preserving initialization
- Foundation V2
- V2-B direct path, embedding, and FiLM path
- critic architecture and support protocol
- GAE lambda = 0.95
- repaired PPO/action-logprob semantics
- objective definitions
- seed 73001
- 75-update authority screen
- no semantic threshold changes
- no router-temperature, expert-count, depth, or learning-rate tuning

## Predeclared authority / specialization gates

The branch required all of the following:

- heavy-preference routing pairwise L2 mean >= 0.02
- max router deviation from uniform >= 0.02
- expert residual max norm > 1e-3
- mean expert-output diversity > 1e-3
- expert gradient-share pairwise distance >= 0.02
- nonzero router gradients
- modular action authority > 1e-4
- masking the modular path reduces preference-conditioned action separation or preference Jacobian by > 1e-4
- single-expert mask effects vary by preference
- Foundation-V2 critic, PPO-ratio, and survival guardrails remain valid

## Result

### Router behavior — FAIL

The learned router remained close to uniform after 75 updates.

Heavy-preference routing matrix:

| Preference | Expert 1 | Expert 2 | Expert 3 | Expert 4 |
|---|---:|---:|---:|---:|
| T-heavy | 0.25058 | 0.25138 | 0.24362 | 0.25442 |
| A-heavy | 0.25153 | 0.25007 | 0.24618 | 0.25221 |
| O-heavy | 0.25254 | 0.25063 | 0.24335 | 0.25348 |
| S-heavy | 0.25187 | 0.24997 | 0.24480 | 0.25336 |

Diagnostics:
- heavy-preference routing pairwise L2 mean = **0.00257**
- predeclared threshold = **0.02**
- max deviation from uniform = **0.00665**
- predeclared threshold = **0.02**
- routing entropy remains approximately **1.3862**, close to log(4) = 1.38629

Therefore the router does not establish meaningful preference-dependent expert allocation.

### Experts are active — PASS

The private residual experts are not dead.

At the final checkpoint:
- maximum expert residual norm = **1.2653**
- expert residual norm CV = **0.2224**
- mean pairwise expert-output diversity = **0.5418**

Training gradients also reached every expert.

Maximum expert-gradient norms:
- Expert 1: **0.2743**
- Expert 2: **0.3321**
- Expert 3: **0.2101**
- Expert 4: **0.3186**

Thus the failure is not caused by zero-output experts or an inactive modular path.

### Gradient specialization — weakly present

Read-only objective-heavy gradient-share probes give a mean pairwise expert-share distance of **0.02408**, slightly above the predeclared 0.02 threshold.

Gradient shares:

| Preference | Expert 1 | Expert 2 | Expert 3 | Expert 4 |
|---|---:|---:|---:|---:|
| T-heavy | 0.2571 | 0.2834 | 0.1747 | 0.2848 |
| A-heavy | 0.2576 | 0.2752 | 0.1932 | 0.2740 |
| O-heavy | 0.2724 | 0.2679 | 0.1832 | 0.2765 |
| S-heavy | 0.2436 | 0.2761 | 0.1906 | 0.2897 |

This indicates some objective-dependent gradient structure, but it does not translate into preference-dependent routing.

### Router receives gradients — PASS

The router is trainable and receives nonzero gradients:
- max router weight-gradient norm during training = **0.00703**
- median router weight-gradient norm = **0.00244**
- objective-heavy router-gradient norms remain nonzero

Therefore the near-uniform routing cannot be attributed to a disconnected router.

### Modular action authority — active but not preference-specializing

The modular residual has a large causal effect on actions:
- mean modular action authority = **0.2435**

Single-expert masking effects are nonzero and show small preference dependence:
- mean per-expert preference standard deviation = **0.00122**

However, the complete modular path does **not** increase preference-conditioned action authority.

With the modular path active:
- pairwise action separation = **0.07476**
- preference-Jacobian Frobenius norm = **0.03858**

With the modular path masked back to the current V2-B path:
- pairwise action separation = **0.08187**
- preference-Jacobian Frobenius norm = **0.04412**

Therefore the modular path **reduces**, rather than increases, both existing preference-conditioned action separation and local preference sensitivity.

This fails the predeclared causal authority gate.

## Foundation-V2 guardrails — PASS

The architecture change does not destabilize the validated training foundation.

Final critic audit:
- early EV mean = **0.4652**
- early negative fraction = **0.15**
- late EV mean = **0.2523**
- late negative fraction = **0.15**
- combined negative fraction = **0.15**

Optimization / survival:
- max PPO ratio error = **3.05e-05**
- last-10-update termination fraction = **0.0**

Thus V2-C1 fails because of modular specialization/authority, not because Foundation V2 collapsed.

## Interpretation

V2-C produces active private residual experts, but the router does not meaningfully allocate them by preference.

The learned architecture behaves approximately as:

> V2-B + a nearly uniform mixture of four learned residual transformations

rather than:

> V2-B + preference-gated private parameter subspaces.

This distinction is decisive for the branch hypothesis.

Although expert gradients show small objective-dependent differences, the preference router remains near maximum-entropy uniform routing and the combined modular path reduces the preference separation already present in V2-B.

Therefore the experiment does not establish the mechanism required before semantic evaluation:

> **partial preference-gated parameter isolation was not realized.**

## Decision

**V2-C1 FAIL.**

Per the predeclared stop rule:

- V2-C2 semantic gate is **NOT AUTHORIZED**
- no semantic endpoint claim is made from V2-C1
- no router-temperature tuning
- no router regularizer
- no expert-count tuning
- no expert-depth tuning
- no learning-rate tuning
- no hard objective-to-expert assignment
- no sparse/top-k routing modification

The V2-C branch is closed at the authority/specialization stage.

The final thesis reference therefore returns to:

> **V2-B + Foundation V2 + GAE lambda = 0.95 + no retention intervention**

## Scientific implication

The result does **not** show that modular policies or MoE architectures are ineffective in general.

It shows that the minimal function-preserving deterministic softmax-routing adaptation tested here did not spontaneously develop the preference-dependent routing required to test the intended parameter-isolation hypothesis.

The architecture acquired additional active capacity, but not the desired preference-conditioned allocation of that capacity.

Primary artifacts:
- `scripts/rl/v2c1_modular_authority_screen.py`
- `runs/v2c1_modular_authority-2026-09-24/v2c1_report.json`
- `runs/v2c1_modular_authority-2026-09-24/model_75.pt`
- `docs/contracts/preference_architectures/v2c-contract.md`
