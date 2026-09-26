# V2-B Finite Gradient-to-Behavior Realization Verdict

Status: FROZEN — LOCAL ANGULAR CREDIT CAN BE REALIZED BY A FINITE SHARED-POLICY UPDATE

Date: 2026-09-24

## Question

At the lambda=1 u75 checkpoint where Angular semantics have collapsed, does a controlled finite step along the measured Angular gradient produce the expected closed-loop Angular behavior, or does the local gradient fail when realized through the shared network and nonlinear dynamics?

## Design

Base policy: lambda=1 continuous-Adam u75 checkpoint.

A-heavy gradient was estimated from four fresh matched on-policy batches at lambda=1.

Two intervention directions were tested:
- A-only descent direction
- A-heavy scalarized combined-gradient descent direction

Step norm was normalized by the actual mean actor parameter delta during u51-u75 of the real lambda=1 training run:
- nominal actor step norm = 0.047012

Scales:
- 0
- 0.25x
- 0.5x
- 1.0x
- 2.0x nominal step

No further training or optimizer updates were performed.

Measured:
- same-batch A surrogate
- fixed-state action displacement
- 16-step A-heavy rollout
- full 64-step matched endpoint suite T/A/O/S/C over four resets
- absolute A-heavy and center |omega_xy|
- survival

Important caveat:
The combined direction here is the raw scalarized combined-gradient descent direction on the frozen A-heavy batch. It is not the exact Adam-preconditioned update direction because full optimizer moment tensors were not serialized.

## Local surrogate response

A-only direction:
- baseline A weighted loss: 0.40770
- 0.25x: 0.38634
- 0.5x: 0.38046
- 1.0x: 0.38510
- 2.0x: 0.40065

The local surrogate improves strongly for small/moderate finite steps, with the best tested loss near 0.5x nominal.

The A-only and scalarized combined descent directions are highly aligned on this A-heavy batch:
- cosine = 0.97996

## Angular endpoint realization

A-only finite step:

| scale | A objective correctness | A physical correctness |
|---:|---:|---:|
| 0 | 0.00 | 0.00 |
| 0.25x | 0.25 | 0.50 |
| 0.5x | **0.75** | **0.75** |
| 1.0x | 0.50 | 0.75 |
| 2.0x | **0.75** | **0.75** |

Thus a single finite A-only update can recover the Angular endpoint from complete failure to the passing directional regime.

## Absolute physical Angular response

The endpoint recovery is not an artifact caused only by worsening the center reference.

Matched 64-step absolute A-heavy |omega_xy|:
- base: 1.03178
- A-only 0.5x: 1.01702
- A-only 1.0x: **1.00252**
- A-only 2.0x: 1.01549

For 1.0x, the center is 1.00923, so the A-heavy policy is both absolutely better than baseline and better than its matched center.

Therefore the finite Angular step produces a genuine closed-loop physical improvement.

## Short-horizon response

16-step A-heavy |omega_xy| changes only modestly:
- base: 1.06830
- 0.25x: 1.06242
- 0.5x: 1.06699
- 1.0x: 1.06776
- 2.0x: 1.07465

The strongest semantic improvement emerges over the longer 64-step closed-loop rollout rather than as a monotonic immediate short-horizon effect.

## Combined-gradient control

The raw A-heavy scalarized combined-gradient direction can also recover Angular semantics at some scales:
- 0.25x: A = 0.75 objective / 0.50 physical
- 2.0x: A = 0.75 / 0.75

But response is strongly non-monotonic:
- 0.5x: A = 0.25 / 0.25
- 1.0x: A = 0.50 / 0.25

This shows that finite behavior realization depends on step scale and direction composition even when the local surrogate direction is valid.

## Cross-objective collateral

A-only 0.5x yields:
- T: 0.75 / 0.75
- A: 0.75 / 0.75
- O: 0.25 / 0.25
- S: 0.50 / 0.50
- survival: 1.0

So recovering Angular does not simultaneously solve the 4D semantic problem. The intervention redistributes endpoint semantics across objectives.

## Causal interpretation

The key rejected hypothesis is:
> A locally valid Angular gradient cannot survive finite shared-network realization into closed-loop behavior.

That hypothesis is rejected because a controlled finite A-gradient step genuinely improves absolute Angular dynamics and recovers the A endpoint.

Supported instead:
> The u75 policy manifold still contains a usable local direction for Angular recovery. The late training failure arises from how many valid local updates are sequenced, mixed across objectives/preferences, preconditioned, and accumulated over the non-stationary training path—not from the absence of a realizable Angular direction at the final policy.

## Causal status

- Local Angular gradient availability: ESTABLISHED.
- Static joint-gradient correctness: RETAINED.
- Finite A-gradient step can improve local surrogate: ESTABLISHED.
- Finite A-gradient step can improve absolute closed-loop Angular dynamics: ESTABLISHED.
- Finite A-gradient step can recover A endpoint semantics: ESTABLISHED.
- Generic gradient-to-behavior realization failure: REJECTED.
- One-step finite update as sufficient explanation of late collapse: REJECTED.
- Multi-update sequence/composition/path dependence: PRIMARY REMAINING LOCUS.
- Cross-objective collateral from a successful A recovery step: ESTABLISHED.

## Decision

Do not reopen architecture, lambda, critic, learning-rate, optimizer-reset, or visitation-only branches.

The next causal question, if investigation continues, is no longer whether a single valid gradient can produce valid behavior. It can.

The remaining question is:
> Why does a sequence of individually valid finite updates fail to preserve previously acquired semantic endpoints when objectives/preferences are optimized jointly?

A clean next diagnostic would compare short controlled update sequences with identical starting parameters and matched batches:
- repeated A-only updates
- repeated scalarized mixed-objective updates
- alternating objective-specific updates

while evaluating semantic endpoints after each update and keeping visitation externally matched where possible. This would isolate update-sequence/composition effects from one-step realizability.