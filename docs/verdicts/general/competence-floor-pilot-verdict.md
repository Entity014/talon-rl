# Competence-Floor Paired Short-Pilot Verdict

Status: **FROZEN — PILOT REJECTED; COMPETENCE-FLOOR BRANCH CLOSED**

Date: 2026-09-24

## Question

Does the max-min direction that repairs per-update first-order objective geometry translate into semantic competence retention / accumulation over a short winner-rotation interval?

## Design

Both arms restart from the exact same frozen V2-B + Foundation V2 + lambda=.95 u50 weights.

Because the historical checkpoint does not store Adam state, both arms use fresh Adam state. Everything else is paired:
- same 25 update horizon (global u51..u75)
- same preference schedule
- same reset/support seed schedule
- same critic support / refresh protocol
- same actor optimizer type/lr and gradient clip
- same repaired PPO action/log-probability semantics

Arms:
- control: current weighted MORL PPO direction
- floor: normalized per-objective max-min direction, rescaled to the exact current mixed-gradient norm

No semantic PASS/FAIL signal or past behavior target enters the floor direction.

## Optimization-side result

Both arms preserve Foundation V2.

Control:
- max PPO ratio error = **2.29e-5**
- last-10 termination fraction = **0.0**
- mean parameter-step norm = **0.07535**
- final early critic EV = **0.5989**
- final late critic EV = **0.1583**
- final combined critic negative fraction = **0.10**

Floor:
- max PPO ratio error = **1.53e-5**
- last-10 termination fraction = **0.0**
- mean parameter-step norm = **0.07673**
- final early critic EV = **0.4847**
- final late critic EV = **0.1243**
- final combined critic negative fraction = **0.15**

The treatment successfully maintains the intended local geometry throughout training:
- mean mixed worst normalized gain = **-0.2935**
- mean floor worst normalized gain = **+0.2506**
- all four predicted objective gains are non-negative on **25/25** treatment updates
- mean cosine(floor, mixed) = **0.4568**
- mean floor/mixed gradient-norm ratio = **1.00000000**

Thus the optimization mechanism works exactly as intended and does not collapse step size or Foundation V2.

## Semantic path

Frozen starting checkpoint for both arms:
- step 0: **Orientation PASS only**

### Paired control

| paired step | endpoint PASS axes |
|---:|---|
| 0 | O |
| 5 | A, O |
| 10 | none |
| 15 | none |
| 20 | none |
| 25 | none |

The paired control therefore reproduces the expected non-stationary / forgetting behavior after a brief A+O overlap.

### Max-min floor arm

Completed frozen evaluations before the predeclared gate became mathematically impossible:

| paired step | endpoint PASS axes |
|---:|---|
| 0 | O |
| 5 | none |
| 10 | none |
| 15 | none |
| 20 | none |

Selected endpoint fractions in the floor arm:
- step 5: O = 0.50 objective / 0.50 physical
- step 10: O = 0.50 / 0.50
- step 15: O = 0.25 / 0.25
- step 20: O = 0.50 / 0.50

No endpoint axis passes at any completed post-start floor checkpoint.

## Decisive predeclared-gate failure

The pilot contract requires Orientation, which is PASS at start, to remain PASS at >=3/5 post-start checkpoints.

After completed floor evaluations at steps 5, 10, 15, and 20:
- Orientation retained-PASS count = **0/4**

Only one post-start checkpoint (step 25) remains. Therefore the maximum possible final Orientation retained count is **1/5**, which is strictly below the required **3/5**.

The pilot PASS gate is therefore mathematically impossible to satisfy regardless of the unevaluated final endpoint or continuum result.

Per the bounded stop rule, the remaining floor step-25 semantic evaluation and final continuum evaluation were stopped rather than spending additional compute on a branch that could no longer pass.

## Interpretation

This separates two layers cleanly:

1. **Local optimization geometry**
   - weighted PPO often gives one objective negative first-order improvement
   - max-min combination repairs this robustly
   - worst local gain becomes positive without step collapse

2. **Closed-loop semantic competence**
   - the repaired first-order geometry does not preserve the already-acquired Orientation semantic competence
   - Orientation is lost immediately by the first evaluated post-start checkpoint and remains below PASS through step 20
   - no endpoint competence accumulates despite non-negative predicted per-objective gains on every treatment update

Therefore semantic forgetting is not explained solely by instantaneous negative per-objective PPO gradient projections.

A stronger competence-floor method would need a competence signal whose first-order geometry is more directly coupled to closed-loop semantic outcome, rather than merely balancing the existing per-objective PPO gradients.

## Decision

**COMPETENCE-FLOOR PILOT REJECTED.**

Per the predeclared stop rule:
- no eta / softmax-temperature tuning
- no simplex-resolution tuning
- no optimizer / learning-rate tuning
- no architecture change
- no retention / replay addition
- no semantic-threshold changes
- no multi-seed confirmation

The competence-floor / max-min branch is closed in its tested formulation.

The final reference remains:

> **V2-B + Foundation V2 + GAE lambda=.95 + no retention intervention**

## Scientific implication

A max-min actor-gradient combination can eliminate locally negative objective-update directions while preserving update magnitude, yet this alone is insufficient to prevent behavior-level semantic forgetting.

This strengthens the thesis conclusion that the observed winner rotation is a closed-loop, path-dependent semantic phenomenon not fully characterized by static or one-step gradient geometry.

Primary artifacts:
- `docs/contracts/preference_architectures/competence-floor-pilot-contract.md`
- `scripts/rl/v2b_competence_floor_paired_pilot.py`
- `runs/v2b_competence_floor_paired_pilot-2026-09-24/paired_pilot_report.json`
- `runs/v2b_competence_floor_semantic_path-2026-09-24/control_*/endpoint_report.json`
- `runs/v2b_competence_floor_semantic_path-2026-09-24/floor_{0,5,10,15,20}/endpoint_report.json`
