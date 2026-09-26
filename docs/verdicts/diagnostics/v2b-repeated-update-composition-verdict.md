# V2-B Repeated-Update Composition / Semantic-Retention Verdict

Status: FROZEN — SEMANTIC COMPETENCE ROTATES UNDER SUCCESSIVE SHARED-POLICY UPDATES; BOTH WITHIN-OBJECTIVE PATH INSTABILITY AND CROSS-OBJECTIVE FORGETTING ARE PRESENT

Date: 2026-09-24

## Question

Each local finite update can be valid and behaviorally realizable. Why do semantic endpoints disappear when updates are repeated and objectives are optimized jointly?

## Controlled design

All arms start from the exact same lambda=1 u75 policy.

Step size is fixed at 0.5x the actual late-training nominal actor parameter displacement:
- nominal late actor delta = 0.047012
- diagnostic finite step = 0.023506

Every update uses a fresh on-policy batch with matched reset seed across arms. No Adam is used; each step is a normalized finite descent step so optimizer memory is removed from this gate.

Three 8-step arms:
- A-only: A, A, A, A, A, A, A, A
- Mixed: scalarized T/A/O/S update every step
- Alternating: A -> O -> S -> T -> A -> O -> S -> T

After every update:
- matched endpoint T/A/O/S/C evaluation over 4 suites
- semantic score R_i = 0.5(objective-correct fraction + physical-correct fraction)
- endpoint pass status
- pairwise preference-conditioned action separation
- cumulative parameter displacement
- cosine to previous update direction
- forgetting F_{i<-j} = R_i(before update j) - R_i(after update j)

Baseline at theta0:
- T score 0.375, fail
- A score 0.000, fail
- O score 0.500, fail
- S score 0.750, PASS

## A-only path: same objective is not behaviorally monotonic

A semantic scores across repeated A-only updates:

| update | A score | A pass | update cosine vs previous |
|---:|---:|:---:|---:|
| 0 | 0.000 | no | - |
| 1 | **1.000** | **yes** | - |
| 2 | 0.375 | no | -0.039 |
| 3 | 0.500 | no | -0.013 |
| 4 | 0.375 | no | +0.041 |
| 5 | **0.000** | no | -0.108 |
| 6 | 0.625 | no | +0.030 |
| 7 | 0.125 | no | -0.089 |
| 8 | 0.250 | no | -0.040 |

Key result:
> Repeated A-only optimization does not preserve the A endpoint even though the first A step fully recovers it.

Successive fresh A gradients are almost orthogonal or mildly anti-aligned in parameter space. Thus a single objective's on-policy descent direction itself rotates strongly as the policy moves.

A-only pass frequency over 8 post-update checkpoints:
- T: 2/8
- A: 1/8
- O: 2/8
- S: 1/8

This rejects a pure cross-objective-only explanation.

## Alternating path: direct cross-objective forgetting is also real

Alternating sequence:
- u1 A: A score 0 -> 1.00, A PASS
- u2 O: A 1.00 -> 0.25
- u3 S: A 0.25 -> 0.50
- u4 T: A 0.50 -> 0.625
- u5 A: A 0.625 -> 0.75, A PASS
- u6 O: A 0.75 -> 0.50
- u7 S: A 0.50 -> 0.375
- u8 T: A 0.375 -> 0.625

Average forgetting matrix by causing update label (positive = semantic loss):

| causing update | T | A | O | S |
|---|---:|---:|---:|---:|
| A | +0.375 | **-0.562** | -0.125 | -0.188 |
| O | **-0.500** | **+0.500** | +0.188 | -0.125 |
| S | +0.125 | -0.062 | -0.062 | +0.188 |
| T | 0.000 | -0.188 | 0.000 | **+0.375** |

Interpretation:
- A-updates reliably restore A on average.
- O-updates strongly erase A competence (mean F_A<-O = +0.50), reproduced in both cycles.
- T-updates strongly erase S competence (mean F_S<-T = +0.375).
- O and S updates can even reduce their own behavior-level semantic score, showing that local objective descent does not imply monotonic endpoint behavior at every point on the path.

This is direct evidence of cross-objective semantic interference / forgetting, but it coexists with within-objective instability.

## Mixed scalarized path: competence is redistributed, not accumulated

Mixed-arm endpoint passes by update:
- u1: none
- u2: none
- u3: O and S
- u4: none
- u5: O
- u6: none
- u7: none
- u8: A

No objective retains a passing endpoint consistently.

Mean semantic scores across the 8 mixed updates:
- T: 0.359
- A: 0.438
- O: 0.516
- S: 0.500

Pass counts:
- T: 0/8
- A: 1/8
- O: 2/8
- S: 1/8

Thus mixed updates do not simply suppress all objectives. They repeatedly move semantic competence among axes.

## Conditioning authority is preserved

Mean pairwise fixed-state preference separation remains nearly constant in all arms (~0.060-0.063) while semantic endpoints appear and disappear.

Therefore the retention failure is not caused by loss of preference sensitivity or collapse of the conditioning mechanism.

## Parameter-path observations

Cumulative displacement from theta0 grows smoothly to roughly 0.061-0.066 by update 8 in all arms.

Yet semantic scores oscillate sharply instead of changing monotonically with distance.

Successive update-direction cosines are generally near zero and frequently negative, including within the A-only arm.

This indicates a rapidly rotating local optimization field on the shared policy manifold.

## Causal interpretation

The evidence does not support the simple statement:
> Valid A competence is learned, then only other objectives catastrophically overwrite it.

The stronger supported statement is:
> Semantic competence is path-dependent and non-stationary under successive shared-policy updates. Cross-objective updates can directly erase previously acquired competence, but even repeated optimization of one objective can rotate into regions where its own behavior-level semantics disappear. Multi-objective training therefore experiences both within-objective path instability and cross-objective semantic interference.

This explains the earlier training observations:
- locally correct gradients can exist at every checkpoint
- one finite A step can recover Angular behavior
- A can pass at u50 and fail at u75
- no static gradient defect is required
- semantic winners can rotate over training even when preference authority remains intact

## Causal status

- One-step gradient correctness: CLOSED.
- Finite-step behavior realizability: CLOSED / passes.
- Loss of conditioning authority: REJECTED.
- Pure cross-objective-only forgetting: REJECTED as complete explanation.
- Within-objective repeated-update instability: ESTABLISHED.
- Cross-objective semantic forgetting: ESTABLISHED.
- Mixed-objective competence redistribution: ESTABLISHED.
- Rapid rotation of successive local update directions: ESTABLISHED.
- Stable semantic retention under the current shared-policy optimization rule: REJECTED.

## Thesis-level finding

> The central limitation is not the absence of valid objective gradients, but the lack of semantic retention under a rapidly changing shared-policy optimization field. Objective-specific competence can be acquired by a valid finite update, lost by subsequent updates of other objectives, and even lost under further updates of the same objective as the policy moves to a new local regime.

## Decision

Do not reopen reward, critic, lambda, learning rate, architecture capacity, optimizer-memory, or visitation-only branches.

The next method question is now justified and narrow:
> Should the optimization rule explicitly preserve previously acquired semantic competence across updates?

Evidence-based candidate classes now include semantic-retention / interference-control mechanisms rather than additional conditioning capacity. Before selecting a final method, a minimal next gate should compare one retention-preserving update rule against the same shared-policy baseline, using this exact semantic-retention trace as the primary contract.

Examples of method families that are now experimentally justified to consider (not yet selected):
- gradient projection / conflict-aware updates against retained semantic reference gradients
- rehearsal/reference-state constraints that preserve endpoint behavior
- trust-region constraints defined on preference-conditioned action/semantic change
- objective-specific update buffering or constrained sequential updates

Any intervention should be judged by retention first, not merely by instantaneous gradient cosine or endpoint performance at the final checkpoint.