# V2-B Preference-Conditioned Action Trust-Region Gate Verdict

Status: FROZEN — BEHAVIOR-SPACE ACTION RETENTION HAS STRONGER SIGNAL THAN GRADIENT RETENTION, BUT STATIC ACTION ANCHORS ARE NOT SUFFICIENT AND BECOME PLASTICITY-LIMITING

Date: 2026-09-24

## Question

Can semantic competence be retained by constraining preference-conditioned actions on fixed semantic reference states after an axis has reached PASS?

## Final clean design (v3)

Base policy: lambda=1 V2-B u75 checkpoint.

Control and treatment use:
- same theta0,
- same fresh mixed on-policy batches,
- same lambda/objectives/critic/foundation,
- same semantic evaluator,
- same effective parameter-step norm at every update.

Treatment only:
- when axis i first PASSes, snapshot fixed semantic anchor states S_i and deterministic reference actions a_i_ref = pi_ref(S_i, w_i),
- propose the same mixed update direction as baseline,
- correct the finite parameter displacement to reduce actual deterministic action drift on all active anchors,
- if no full-norm feasible point is found, backtrack step magnitude until every active anchor satisfies the action trust radius.

Trust radius:
- 0.25 x mean pairwise preference-conditioned action separation on that anchor-state set.

Crucially, control uses the exact same backtracked step norm as treatment at each update. Therefore any treatment-control difference is not explainable merely by smaller updates.

## Constraint enforcement

The final v3 treatment satisfies all active action trust constraints:
- every final mean anchor drift / radius <= 1.0.

Examples:
- u4: S 0.997, A 0.622, O 0.574
- u5: S 0.996, A 0.651, O 0.600, T 0.098
- u8: S 0.995, A 0.589, O 0.553, T 0.296

Thus the final verdict is not confounded by failed trust-region enforcement.

## Competence acquisition

Treatment activation times:
- S: baseline checkpoint 0
- A: u3
- O: u3
- T: u4

All four semantic axes are therefore acquired at least once under the treatment.

Control under the same matched step schedule never acquires T or A.

## Endpoint-PASS event count

Across 8 post-update checkpoints:
- magnitude-matched control: 5 PASS events
- action-trust treatment: 10 PASS events

The behavior-space retention constraint therefore has a substantial positive effect on the amount of semantic competence present along the optimization path.

## Forgetting

Max forgetting from historical best semantic score:

| axis | matched control | action trust |
|---|---:|---:|
| T | 0.375 | 0.375 |
| A | 0.500 | **0.375** |
| O | **0.500** | 0.625 |
| S | 0.750 | **0.375** |

Action anchoring improves A and S worst-case forgetting, leaves T unchanged, and worsens O worst-case forgetting.

Thus retention benefit is real but not uniform across objectives.

## Retained-PASS fractions after first PASS

| axis | matched control | action trust |
|---|---:|---:|
| T | never acquired | 0.25 |
| A | never acquired | 0.40 |
| O | 0.20 | **0.40** |
| S | 0.375 | 0.25 |

Treatment learns more axes and retains O more often, but does not keep every acquired endpoint continuously active.

## Plasticity cost

Once T/A/O/S anchors are all active, the feasible trust region becomes extremely restrictive.

Effective step ratios relative to the nominal 0.5x diagnostic step:
- u5: 0.03125
- u6: 0.0078125
- u7: 0.125
- u8: 0.0078125

At u6/u8 the update is only ~0.8% of the nominal diagnostic step.

Because the control uses these same tiny step magnitudes, the treatment-control comparison remains fair; however, the method itself incurs a major stability-plasticity trade-off.

## Decisive counterexample: action snapshots are preserved but semantic endpoints can still disappear

By construction, retained action anchors remain inside their trust radius.

Nevertheless:
- O is retained as an anchor from u3 onward,
- O endpoint later fails and finishes at semantic score 0.25 at u8,
- A and S also lose PASS status at multiple later checkpoints despite their anchor actions remaining constrained.

Therefore:
> preserving deterministic actions on a finite fixed reference-state set is not sufficient to preserve full closed-loop semantic competence.

This directly supports the interpretation that endpoint semantics depend on trajectory/state-distribution evolution beyond static anchor snapshots.

## Comparison to reference-gradient retention

Reference-gradient projection previously increased PASS events from 4 to 8 but could satisfy all first-order constraints while semantics still disappeared.

Action-space trust retention is stronger:
- it directly constrains realized policy behavior,
- increases PASS events further (5 to 10 under magnitude-matched control),
- improves A/S max forgetting.

But the same structural limitation remains at a higher level:
- local/fixed reference preservation does not guarantee trajectory-level semantic retention.

## Causal status

- Retention-method direction: STRONGLY SUPPORTED.
- Gradient-space retention alone: INSUFFICIENT.
- Static preference-conditioned action retention: BENEFICIAL.
- Static action anchors sufficient for full semantic retention: REJECTED.
- Improvement explainable by smaller steps alone: REJECTED by magnitude-matched control.
- Stability-plasticity trade-off under multi-axis action anchoring: ESTABLISHED.
- Full semantic competence depends on trajectory-level behavior beyond fixed state-action anchors: SUPPORTED.

## Decision

Do not promote static action-anchor trust region as the final method.

The next method candidate should preserve a trajectory-level semantic object rather than individual fixed-state actions.

Most direct next candidate:
- semantic reference rollout / trajectory trust region,
- retain short matched reference trajectories for axes that have reached PASS,
- constrain degradation of axis-specific physical/return semantics over those short rollouts rather than matching every action exactly,
- allow action adaptation as long as semantic rollout performance is retained.

This is less likely to freeze plasticity than action cloning, because multiple action sequences may be accepted if they preserve the semantic outcome.

Primary next gate should remain the same repeated-update protocol with magnitude-matched control, and should judge max forgetting, PASS retention, acquisition of new axes, effective step size, and survival.