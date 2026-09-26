# Competence-Floor / Max-Min Optimization Verdict

Status: **FROZEN — FIXED-POLICY GEOMETRY PASS; PAIRED TRAINING PILOT FAIL; BRANCH CLOSED**

Date: 2026-09-24

## Question

Can replacing the current weighted-average PPO update direction with a
same-norm max-min / competence-floor direction prevent semantic winner rotation
by ensuring non-negative local first-order improvement for every objective?

This branch begins only after architecture and retention searches were closed.
Its motivation is the V2-K result: genuine preference-private residual
subspaces existed, yet endpoint competence still rotated rather than
accumulating.

## Stage F0 — fixed-policy max-min direction audit

Frozen checkpoints:
- V2-B + Foundation V2 + lambda=.95, update 50
- V2-B + Foundation V2 + lambda=.95, update 75

Each checkpoint was evaluated on four fresh exact training-style mixed
preference batches.

For each objective i, the unscalarized PPO ascent gradient q_i was normalized:

    u_i = q_i / ||q_i||

The diagnostic candidate solved:

    max_alpha min_i u_i^T normalize(sum_j alpha_j u_j)

with alpha on the 4-objective simplex, then rescaled the candidate direction to
the exact norm of the current weighted PPO direction.

### F0 result — PASS

Across 8 checkpoint x seed cases:

- mixed worst normalized predicted gain:
  - mean = **-0.2896**
  - minimum = **-0.5546**

- max-min floor worst normalized predicted gain:
  - mean = **+0.2791**
  - minimum = **+0.2087**

- mean worst-gain improvement = **+0.5687**
- worst gain improved in **8/8** cases
- mixed direction had a negative objective in **7/8** cases
- floor direction reduced negative-axis count in **7/7 conflict cases**
- floor direction gave all four objectives non-negative predicted gain in **8/8** cases
- mean cosine(floor, mixed) = **0.4343**
- minimum cosine(floor, mixed) = **0.1953**
- floor/mixed gradient-norm ratio = **1.00000002 mean**
- maximum norm-ratio error is below 1e-6
- PPO ratio invariant passed

Therefore the fixed-policy geometry clearly supports the competence-floor
hypothesis at first order.

A paired short training pilot was authorized.

## Stage F1 — paired u50 -> u75 training pilot

Both arms started from the exact same frozen u50 model weights.

Because the stored checkpoint does not contain Adam optimizer state, both arms
used a fresh paired Adam restart. The historical u75 checkpoint was context
only, not the paired control.

Frozen across both arms:
- V2-B architecture
- Foundation V2
- GAE lambda=.95
- objective definitions
- rollout horizon H32
- matched global update preference schedule 51..75
- matched reset / support seeds
- historical critic anchor specs
- Adam lr=1e-3
- grad clip=1.0
- semantic thresholds/evaluator
- no replay or retention

Arms:
- CONTROL: original weighted MORL PPO gradient
- FLOOR: same-norm max-min gradient from the passed F0 formulation

Duration:
- 25 paired updates
- checkpoints at paired steps 0, 5, 10, 15, 20, 25

### Optimization / Foundation result

Both arms remained optimization-stable.

CONTROL:
- max PPO ratio error = **2.29e-5**
- last-10 termination fraction = **0.0**
- mean parameter-step norm = **0.07535**
- early critic EV = **0.5989**
- late critic EV = **0.1583**
- combined negative critic fraction = **0.10**

FLOOR:
- max PPO ratio error = **1.53e-5**
- last-10 termination fraction = **0.0**
- mean parameter-step norm = **0.07673**
- early critic EV = **0.4847**
- late critic EV = **0.1243**
- combined negative critic fraction = **0.15**

During all 25 floor updates:
- mean mixed worst predicted gain = **-0.2935**
- mean floor worst predicted gain = **+0.2506**
- floor all-objective non-negative predicted-gain fraction = **1.0**
- mean cosine(floor, mixed) = **0.4568**
- mean floor/mixed raw gradient norm ratio = **1.0**

Thus the intervention preserved the intended first-order geometry throughout
training and did not collapse update magnitude.

## Semantic endpoint timeline — FAIL

Frozen matched-reset endpoint evaluation was applied at paired steps
0/5/10/15/20/25.

CONTROL endpoint PASS axes:
- step 0: **O**
- step 5: **A, O**
- step 10: none
- step 15: none
- step 20: none
- step 25: none

FLOOR endpoint PASS axes:
- step 0: **O**
- step 5: none
- step 10: none
- step 15: none
- step 20: none
- step 25: **A**

Predeclared accumulation metrics:
- floor post-start total PASS events = **1**
- control post-start total PASS events = **2**
- floor event advantage = **-1**
- floor maximum simultaneous PASS count = **1**
- floor Orientation retained checkpoints after start = **0/5**
- floor final PASS count = **1**
- control final PASS count = **0**

The floor arm therefore fails the central semantic goal:

> it does not preserve the initially acquired Orientation competence and does
> not produce simultaneous multi-axis competence.

Instead, the winner rotates from Orientation at start to Angular at the final
checkpoint.

## Final full semantic contract

### Paired control, step 25

Endpoint PASS axes:
- none

Continuum:
- monotonicity = **0.60417**
- endpoint-between = **0.48611**

Guardrails:
- min survival = **1.0**
- critic H32 EV mean = **0.4730**
- critic negative fraction = **0.05**

### Competence-floor arm, step 25

Endpoint PASS axes:
- **Angular only**

Angular correctness:
- objective = **1.00**
- physical = **1.00**

Orientation remains incomplete:
- objective = **0.75**
- physical = **0.50**

Continuum:
- monotonicity = **0.57292**
- endpoint-between = **0.43750**

Both continuum metrics are lower than the paired control.

Full-evaluator guardrails:
- min survival = **0.875**
- critic H32 EV mean = **0.3321**
- critic negative fraction = **0.06875**

Thus the full semantic evaluator also detects a survival regression not visible
in the short training-window termination diagnostic.

## Predeclared pilot gate result

1. Foundation/PPO guardrails during training: **PASS**
2. Floor has >=2 simultaneous endpoint PASS axes post-start: **FAIL**
3. Floor has at least +2 PASS events over control: **FAIL**
4. Initial Orientation retained >=3/5 post-start checkpoints: **FAIL (0/5)**
5. Final floor PASS count not lower than control: **PASS (1 vs 0)**
6. Final floor continuum not lower than control on both metrics: **FAIL**

Overall paired pilot verdict: **FAIL**.

## Interpretation

The branch separates two levels of optimization behavior very cleanly.

At the local first-order gradient level, the max-min direction does exactly what
it was designed to do:
- removes negative predicted objective gains,
- improves the worst objective direction strongly,
- keeps essentially the same gradient and parameter-step scale,
- remains meaningfully aligned with the original mixed direction.

However, this local property does **not** translate into closed-loop semantic
retention over successive PPO updates.

The strongest counterexample is Orientation:
- O is semantically PASS at the frozen u50 starting checkpoint;
- every floor update has non-negative predicted local gain for all objectives;
- yet O is not PASS at any of the five post-start floor checkpoints.

Therefore:

> Non-negative first-order improvement of the current per-objective PPO
> surrogates is not sufficient to preserve previously acquired behavior-level
> semantic competence under the current policy/environment dynamics.

This suggests that the semantic forgetting mechanism is not captured by the
instantaneous local PPO objective-gradient floor alone. Higher-order parameter
trajectory effects, state-distribution shift, closed-loop realization, and/or
mismatch between per-objective PPO surrogate improvement and semantic
heavy-vs-center competence remain plausible explanations.

## Decision

**Competence-floor / max-min branch CLOSED.**

Per the predeclared stop rule:
- no eta tuning
- no soft-min temperature tuning
- no simplex-resolution tuning
- no CAGrad/Nash-MTL follow-up training
- no optimizer/lr/lambda change
- no architecture change
- no retention addition
- no semantic-threshold change
- no multi-seed competence-floor confirmation

The final thesis reference remains:

> **V2-B + Foundation V2 + GAE lambda=.95 + no retention intervention**

The max-min direction audit remains a useful negative mechanism result:

> a locally conflict-avoiding update can be geometrically valid yet still fail
> to retain closed-loop semantic competence across training.

## Primary artifacts

- `docs/contracts/preference_architectures/competence-floor-maxmin-contract.md`
- `scripts/rl/v2b_competence_floor_maxmin_audit.py`
- `runs/v2b_competence_floor_maxmin_audit-2026-09-24/competence_floor_maxmin_report.json`
- `docs/contracts/preference_architectures/competence-floor-pilot-contract.md`
- `scripts/rl/v2b_competence_floor_paired_pilot.py`
- `runs/v2b_competence_floor_paired_pilot-2026-09-24/paired_pilot_report.json`
- `scripts/rl/v2b_competence_floor_endpoint_timeline.py`
- `runs/v2b_competence_floor_endpoint_timeline-2026-09-24/endpoint_timeline_report.json`
- `runs/v2b_competence_floor_final_control-2026-09-24/semantic_report.json`
- `runs/v2b_competence_floor_final_floor-2026-09-24/semantic_report.json`
