# Competence-Floor Paired Short-Pilot Contract

Status: **PREDECLARED BEFORE PILOT TRAINING**

Date: 2026-09-24

## Authorization

The fixed-policy max-min audit passed every predeclared diagnostic gate:
- equal gradient-norm budget,
- worst predicted gain improved in 8/8 cases,
- mixed negative-axis conflicts repaired in 7/7 conflict cases,
- floor direction non-negative for all four objectives in 8/8 cases.

A single bounded training pilot is therefore authorized.

## Starting point

Both arms start from the exact same frozen checkpoint:
- V2-B + Foundation V2 + lambda=.95
- `runs/v2b_lambda095_pilot-2026-09-23/model_50.pt`

Frozen semantic context at this checkpoint:
- Orientation endpoint PASS
- Tracking FAIL
- Angular FAIL
- Smoothness FAIL

The historical path later rotates to Angular at u75 while Orientation collapses.

Because the stored checkpoint does not contain Adam optimizer state, this pilot
uses a paired restart:
- both arms receive fresh Adam state,
- both arms start from identical u50 model weights,
- both arms use the same global-update preference schedule (51..75),
- reset/support seeds are matched by global update,
- critic-support and refresh rules are identical.

The historical u75 checkpoint is context only, not the paired control.

## Arms

CONTROL:
- exact existing weighted MORL PPO loss.

FLOOR:
- same per-objective PPO surrogates,
- compute four unscalarized objective ascent gradients,
- solve the deterministic normalized max-min simplex direction exactly as in
  the passed diagnostic audit,
- rescale floor gradient to the exact norm of the current mixed PPO gradient,
- write that gradient into the same actor parameters,
- retain Adam lr=1e-3 and grad clip=1.0.

No competence score, semantic PASS/FAIL label, evaluation result, or past
checkpoint is used to form the floor direction.

## Frozen components

- V2-B architecture
- Foundation V2 critic body/head structure and support refresh protocol
- GAE lambda=.95
- reward/objective definitions and normalization
- transformed-action PPO semantics
- preference schedule
- rollout horizon H32
- actor optimizer type/lr
- gradient clipping
- semantic evaluator and thresholds
- no retention/replay

## Pilot duration

25 paired updates, corresponding to global updates 51..75.

Save checkpoints at paired steps:
- 0, 5, 10, 15, 20, 25

## Online optimization diagnostics

Every update report for both arms:
- PPO ratio error
- actor gradient norm before clipping
- actor parameter step norm
- termination fraction

Floor arm additionally reports:
- mixed worst normalized predicted gain
- floor worst normalized predicted gain
- number of negative axes before/after floor combination
- max-min simplex weights
- cosine(floor,mixed)
- floor/mixed gradient norm ratio

## Semantic evaluation

After training, run the exact frozen matched-reset endpoint contract at steps
0/5/10/15/20/25 for both arms.

Run the full endpoint+continuum frozen semantic contract at final step 25 for
both arms.

## Primary semantic pilot gate

The competence-floor pilot is considered promising only if all of the following
hold:

1. Foundation/PPO guardrails remain valid in both arms.
2. The floor arm shows at least one post-start checkpoint with >=2 simultaneous
   endpoint PASS axes.
3. Across the five post-start endpoint checkpoints, the floor arm has at least
   two more total PASS events than the paired control.
4. Orientation, which is PASS at the frozen u50 start, is retained as PASS at
   >=3/5 post-start checkpoints in the floor arm.
5. At final step 25, floor endpoint PASS count is not lower than control.
6. Final floor continuum monotonicity and endpoint-between are not both lower
   than the paired control.

If these criteria fail, close the competence-floor branch without tuning eta,
softmax temperature, simplex resolution, optimizer, learning rate, lambda,
architecture, retention, or semantic thresholds.

If they pass, only then consider a multi-seed confirmation or a smoother
competence-floor formulation.
