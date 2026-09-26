# V2-K1 Private-Subspace Authority / Diversity Verdict

Status: **FROZEN — V2-K1 PASS; V2-K2 SEMANTIC GATE AUTHORIZED**

Date: 2026-09-24

## Question

Can continuous preference-conditioned coefficients drive genuinely distinct
private residual subspaces without reproducing the common-direction collapse
seen in V2-C and V2-H?

V2-K1 intentionally does **not** evaluate semantic endpoints.

## Frozen contract

- V2-K0 exact function-preserving initialization
- Foundation V2
- complete V2-B direct / embedding / FiLM path
- critic architecture and support protocol
- GAE lambda = 0.95
- repaired PPO/action-logprob semantics
- objective definitions
- seed 73001
- 75-update authority screen
- four unlabeled 128 -> 32 -> 128 private residual modules
- preference coefficient network 4 -> 16 -> 4
- no softmax routing
- no semantic threshold changes
- no module-count, bottleneck, coefficient-network, LR, lambda, entropy, or
  foundation tuning

## Result

**All predeclared K1 criteria pass.**

### Preference -> coefficient authority

Heavy-preference coefficient vectors:

| Preference | c1 | c2 | c3 | c4 |
|---|---:|---:|---:|---:|
| T | -0.03307 | 0.02722 | -0.01808 | 0.01605 |
| A | -0.01246 | 0.03534 | -0.02789 | -0.00450 |
| O | -0.01170 | 0.03839 | -0.02943 | 0.00282 |
| S | -0.01511 | 0.02666 | -0.02135 | 0.01010 |
| C | -0.01841 | 0.03221 | -0.02441 | 0.00606 |

Geometry:
- mean pairwise coefficient distance = **0.02060** >= 0.02
- centered coefficient s2/s1 = **0.3297**
- effective rank at 5% = **3**
- coefficient Jacobian Frobenius norm = **0.04909**
- local coefficient-Jacobian s2/s1 = **0.3211**

Thus the coefficient map is preference-dependent and non-rank-1.

### Preference -> private residual direction diversity

The combined private residuals are not dominated by the near-common direction
observed in V2-H.

Heavy-preference residual cosine matrix:

| | T | A | O | S |
|---|---:|---:|---:|---:|
| T | 1.0000 | 0.8800 | 0.9058 | 0.9691 |
| A | 0.8800 | 1.0000 | 0.9965 | 0.9640 |
| O | 0.9058 | 0.9965 | 1.0000 | 0.9805 |
| S | 0.9691 | 0.9640 | 0.9805 | 1.0000 |

Direction-diversity statistic:
- mean pairwise (1 - cosine) = **0.05069**
- predeclared threshold = **0.02**
- maximum pairwise direction difference = **0.1200**

Pairwise residual-vector separation:
- mean = **0.01958**
- threshold = **0.01**

Centered residual manifold:
- s1 = **0.06754**
- s2 = **0.02588**
- s3 = **0.00894**
- s2/s1 = **0.3832**
- effective rank = **3**

This is materially different from V2-H, whose generated absolute parameter
deltas had mean pairwise (1 - cosine) = 0.00323.

### Preference-dependent private-module allocation

Module contribution shares:

| Preference | M1 | M2 | M3 | M4 |
|---|---:|---:|---:|---:|
| T | 0.3096 | 0.3486 | 0.1765 | 0.1654 |
| A | 0.1315 | 0.5102 | 0.3062 | 0.0521 |
| O | 0.1194 | 0.5363 | 0.3129 | 0.0314 |
| S | 0.1790 | 0.4294 | 0.2615 | 0.1301 |

Diagnostics:
- mean pairwise module-share distance = **0.1891**
- minimum pairwise module-share distance = **0.0361**
- all four modules exceed 0.10 contribution share for at least one heavy
  preference
- maximum observed module share = **0.5363**

Therefore the combination does not collapse to uniform allocation or one
single private module.

### Private modules remain live and diverse

Final private-module output norms are approximately 0.71--0.97 across the
fixed-state probes.

Raw private-module output diversity:
- mean pairwise output distance = **1.2152**
- minimum = **0.7943**
- maximum = **1.6492**

All private modules also receive nonzero training gradients.

Maximum private-module gradient norms:
- M1 = **0.01052**
- M2 = **0.02250**
- M3 = **0.01644**
- M4 = **0.00990**

### Private path has causal action authority

Mean full-vs-private-masked action difference:
- **0.01457** > 1e-4

Preference-conditioned action separation:
- full V2-K = **0.09212**
- private path masked = **0.08741**
- gain from private path = **0.00471**

Action-preference Jacobian:
- full V2-K = **0.04097**
- private path masked = **0.03979**

Both authority metrics therefore increase when the private path is active.

Single-module masking is also preference-specific:
- mean per-module preference standard deviation = **0.001546**

### Trainability

- max coefficient-output gradient norm = **0.1823**
- median coefficient-output gradient norm = **0.02325**
- all four private modules receive persistent nonzero gradients

Thus the observed specialization is not a disconnected diagnostic artifact.

### Foundation V2 guardrails

Final critic audit:
- early EV mean = **0.5120**
- early negative fraction = **0.05**
- late EV mean = **0.2137**
- late negative fraction = **0.10**
- combined negative fraction = **0.075**

Optimization / survival:
- max PPO ratio error = **2.28e-5**
- last-10-update termination fraction = **0.0**

Foundation V2 remains valid.

## Interpretation

V2-K resolves the authority-stage failures of both predecessor branches.

Compared with V2-C:
- private modules are active **and**
- their contribution shares differ strongly by preference.

Compared with V2-H:
- the conditional representation remains multidimensional **and**
- the resulting private residual directions are no longer overwhelmingly
  parallel across preferences.

The combination therefore establishes the mechanism required before semantic
testing:

> preference-conditioned coefficients drive genuinely distinct private
> residual subspaces with measurable causal influence on policy action.

This does not yet establish semantic improvement.

## Decision

**V2-K1 PASS.**

V2-K2 exact semantic accumulation gate is authorized.

K2 must use:
- the frozen endpoint semantic evaluator
- the frozen continuum evaluator
- the same semantic thresholds
- the same evaluation seeds/suites
- Foundation V2
- GAE lambda = 0.95
- no retention intervention

Primary K2 question:

> Do semantic successes accumulate across objectives instead of rotating among
> objectives under continued shared-policy optimization?

No K3 retention/path audit is authorized unless K2 shows meaningful semantic
improvement.

Primary artifacts:
- `talon_rl/v2k_actor_critic.py`
- `scripts/rl/v2k1_private_subspace_screen.py`
- `runs/v2k1_private_subspace-2026-09-24/v2k1_report.json`
- `runs/v2k1_private_subspace-2026-09-24/model_75.pt`
- `docs/contracts/preference_architectures/v2k-contract.md`
