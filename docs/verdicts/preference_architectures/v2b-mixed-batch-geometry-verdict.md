# V2-B Mixed Training-Batch Geometry Verdict

Status: FROZEN — STATIC MIXED-BATCH GEOMETRY DOES NOT EXPLAIN ANGULAR COLLAPSE

Date: 2026-09-24

## Design

Read-only audit on actual paired lambda=.95 and lambda=1.0 training checkpoints at u10/u25/u50/u75.
At every checkpoint, fresh on-policy rollouts used the exact training-style mixed endpoint preference batch (2 envs each for T/A/O/S), with matched reset seeds.

Measured per-objective weighted gradient shares, pairwise cosine, combined-gradient alignment, and combined norm in all actor, shared body, direct preference input, embedding, FiLM, and actor head.

## Main result

Lambda=1.0 changes joint update geometry substantially, especially update magnitude, but does not produce a persistent suppression of Angular credit.

All-actor actual control -> treatment changes:

- u10: A share 0.288 -> 0.238; combined cos(A) 0.658 -> 0.539
- u25: A share 0.209 -> 0.196; combined cos(A) 0.540 -> 0.540
- u50: A share 0.293 -> 0.317; combined cos(A) 0.661 -> 0.709
- u75: A share 0.258 -> 0.255; combined cos(A) 0.537 -> 0.589

Thus early treatment geometry temporarily reduces Angular contribution, but the effect does not persist and reverses by u50/u75 even though final Angular endpoint behavior is worse under lambda=1.

## Tracking / Orientation / Smoothness pattern

Tracking combined-gradient alignment is reduced more consistently under lambda=1:
- u25: 0.126 -> 0.004
- u50: 0.146 -> -0.017
- u75: 0.133 -> -0.039

Orientation alignment increases at u25/u50/u75, consistent with the observed Orientation behavioral improvement.

Smoothness alignment also tends to improve later, consistent with Smoothness becoming the passing endpoint under lambda=1.

## Update magnitude

Lambda=1 substantially increases mixed-batch combined gradient norm:
- u10: ~1.57x
- u25: ~1.87x
- u50: ~1.83x
- u75: ~2.05x

This remains a plausible contributor to optimization-path differences under the fixed Adam learning rate, but it is not equivalent to objective-specific suppression.

## Layer observations

No single layer shows a stable Angular-suppression signature across the whole path.
At u75 the embedding block shifts strongly from Angular toward Orientation (A share delta ~-0.124, O ~+0.110), but this is not mirrored in the total actor gradient and is not temporally consistent enough to establish a causal bottleneck by itself.

## Causal status

- Static A-heavy suppression: rejected.
- Static mixed-training-batch Angular suppression: not supported as persistent cause.
- Lambda=1 update-scale amplification: supported.
- Tracking antagonism under lambda=1: supported more consistently.
- O/S joint-gradient improvement: partially supported and behaviorally consistent.
- Final Angular collapse: remains downstream of instantaneous joint-gradient composition.

## Decision

Do not introduce gradient surgery, vector lambda, or architecture changes from this evidence.

Next gate: semantic checkpoint-path audit at u10/u25/u50/u75 using the exact frozen semantic evaluator. The goal is to locate when Angular semantics actually deteriorate relative to the already-measured gradient path.