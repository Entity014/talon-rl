# Teacher V4 — V4-C2S Smoothness Reformulation Verdict

Status: **FROZEN — no candidate selected. All four smoothness formulations are nearly the same objective as A on these policies (ρ 0.79–0.88 per step, 0.91–0.95 per window), and none meets the eligibility bound against O.**
Date: 2026-09-27
Branch: `v4-c2-semantic-preservation`
Contract: [teacher-v4-c2s-smoothness-reformulation-contract.md](../../contracts/teacher_v4/teacher-v4-c2s-smoothness-reformulation-contract.md) (frozen at `fd78e79`)
Data: `s_candidates_audit.npz` in the 10 primary run directories (`s_candidates_audit.py`, `bb8ddd7`); aggregate `runs/teacher_v4_c2s_smoothness_audit-2026-09-27/aggregate.json` (`s_candidates_aggregate.py`). No snapshots excluded.

## Result (medians over the 10 primary checkpoints)

| candidate | vs A: ρ step | ρ 32-step | off-diag | PC1 snap | vs O: ρ step / win | vs T: ρ step / win | eligible |
|---|---:|---:|---:|---:|---|---|---|
| S0 action rate (current) | 0.843 | 0.916 | 0.138 | 0.955 | 0.212 / 0.287 | −0.05 / −0.03 | no |
| S1 action jerk | 0.806 | 0.912 | 0.169 | 0.847 | 0.194 / 0.269 | −0.04 / −0.01 | no |
| S2 joint-velocity jerk | 0.791 | 0.925 | 0.184 | 0.935 | 0.181 / 0.231 | −0.03 / −0.01 | no |
| S3 torque rate | 0.875 | 0.952 | 0.124 | 0.972 | 0.218 / 0.296 | −0.05 / −0.02 | no |

Eligibility (fixed in advance): |ρ| with T and O must not exceed the V4-C2F
A–O values (0.16 per step, 0.23 per window). Every candidate is uncorrelated
with T. Every candidate slightly exceeds the bound against O: 0.18–0.22
per step, and the closest is S2 at 0.231 vs 0.23 per window. **Selection rule
result: no candidate selected.**

The eligibility bound is not the reason the question fails. Even ignoring
it, the best candidates only modestly reduce redundancy with A (S2 per-step
ρ 0.79 vs S0 0.84; S1 PC1 0.85 vs 0.96), and all stay in the region V4-C2F
called "subsumed".

## Descriptive: not a pooling artifact

Within a single preference, pooled over checkpoints (outside the contract):

| preference | S0 | S1 | S2 | S3 |
|---|---:|---:|---:|---:|
| C | 0.78 | 0.74 | 0.70 | 0.80 |
| T | 0.78 | 0.74 | 0.73 | 0.79 |
| A | 0.75 | 0.65 | 0.60 | 0.78 |
| O | 0.79 | 0.62 | 0.57 | 0.79 |
| S | 0.77 | 0.71 | 0.68 | 0.80 |

The coupling holds within every policy, not only across policies of
different gait intensity. The second-difference forms (S1, S2) are the least
coupled, mainly under A- and O-heavy preferences (0.57–0.65).

## Reading

On this robot, gait and substrate, every measured notion of actuation
smoothness (action rate, action jerk, joint jerk, torque rate) co-varies
strongly with body angular rate. A policy that moves its legs less abruptly
also rotates its body less. Reformulating S in joint or actuator space does
not produce an objective with independent behavioral meaning *on the
behavior these policies produce*.

Limitation (stated in the contract): these policies were trained with S0.
A policy trained on S1 or S2 might decouple them. This audit cannot show
that.

## Options (not decided)

1. **Merge A and S into one stability objective** (three objectives). The
   redundancy now holds across four formulations, not only the original S.
2. **Train-and-re-audit one candidate** (S1 or S2, the least coupled) and
   repeat V4-C2F on the resulting policies. This is the only way to test the
   limitation above. It costs a full V4-C-style training set.
3. **Define S orthogonal to body rotation by construction**, e.g. penalize
   only the component of joint jerk that does not map to trunk angular
   acceleration. This is a more complex formulation that needs its own audit.
