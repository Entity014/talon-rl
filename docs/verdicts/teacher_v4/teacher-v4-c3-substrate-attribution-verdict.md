# Teacher V4 — V4-C3 Substrate Attribution Verdict (H2)

Status: **FROZEN. Registered reading: H2 NOT SUPPORTED (S1 0/6, S2 2/6). The registered explanation attached to that outcome ("the V4 center is on M0's manifold") is false. The V4-C3 policies do not walk. They stand still, so every substrate penalty is lower than M0's.**
Date: 2026-09-28
Branch: `v4-c2-semantic-preservation`
Contract: [teacher-v4-c3-substrate-attribution-contract.md](../../contracts/teacher_v4/teacher-v4-c3-substrate-attribution-contract.md) (frozen at `b35754d`, before measurement)
Runs: `runs/teacher_v4_c3_substrate-2026-09-28/g1_{1,2}_seed{73101,73102,73103}/` (`substrate_attribution.py`), traces in `substrate_traces.npz`.

## Registered result

| run | S1 (≥ 2 substrate terms materially worse at C than M0) | S2 (A⁺ materially degrades ≥ 1 substrate term) |
|---|---|---|
| G1-1 s73101 | ✗ | ✗ |
| G1-1 s73102 | ✗ | ✗ |
| G1-1 s73103 | ✗ | ✗ |
| G1-2 s73101 | ✗ | ✓ (torque) |
| G1-2 s73102 | ✗ | ✗ |
| G1-2 s73103 | ✗ | ✓ (torque) |

S1 0/6 (0/4 without the two fidelity-invalid runs) → **registered reading
"H2 not supported".** In every run, all four substrate terms are materially
**better** at C than under M0, not worse.

## Why the registered reading's explanation is wrong (descriptive)

S1 operationalized "off the locomotion manifold" as "larger substrate
penalties than M0". That missed the case the data show: the V4-C3 policy
family is off the manifold because it does not locomote. Steady window,
from the same snapshots:

| | C (V4-C3, 6 runs) | M0 |
|---|---|---|
| touchdown steps | 0–0.7 % | 13.3–13.7 % |
| all four feet in contact | 92–100 % | 77 % |
| ‖q̈‖ | 7–46 | 330–337 |
| ‖ω_xy‖ (= −S_A) | 0.01–0.13 | 1.00–1.02 |
| tilt (deg) | 0.7–1.6 | 3.8–4.0 |
| `track_lin_vel_xy_exp` (weighted) | 0.298–0.305 | 1.31–1.34 |

With the stock command ranges (vx, vy, ω_z ~ U[−1, 1], σ² = 0.25), the
expected tracking reward of a robot that stands still is 1.5 · 0.194 = 0.292
(linear) and 0.75 · 0.441 = 0.331 (yaw). The C values match this. The V4-C3
center stands.

A⁺ and O⁺ also stand in every run (touchdowns ≤ 2.4 %). Only T⁺ steps, and
only partly: touchdowns 0.1–41 %, and linear tracking 0.30–0.50, still far
below M0's 1.33.

Training logs agree. The normalized T reward per step at iteration 300 is
0.0084–0.0100 for V4-C3 and 0.0071–0.0108 for all 16 V4-C runs. Standing
still gives 0.0072, and M0-level tracking gives 0.0219.

## Consequence for the V4-C3 verdict (descriptive, does not change it)

- **D failure is a floor effect.** At C, ‖ω_xy‖ is already 0.01–0.13
  against 1.0 for a walking gait, so A⁺ has almost nothing to improve. This
  fits the tiny A⁺ − C responses and the B1 failures for A.
- **D critic.** Near-constant D returns around zero leave the critic almost
  no target variance to explain, so EV is unstable and often negative.
  This is consistent with the 6/6 invalid D critics. It was not tested
  directly.
- **The objective formulation itself favors standing.** Normalized objective
  vectors R = [T, D, O] from the steady-window levels: standing ≈ [0.35, 0.00,
  −0.05], M0's gait ≈ [1.09, −0.46, −0.95]. Scalarized, M0's gait minus
  standing is −0.15 to −0.24 at C, −0.32 to −0.36 at A⁺, −0.49 to −0.64 at
  O⁺, and +0.29 to +0.36 only at T⁺ (6/6 runs). Under the T3-B divisors
  (abs-mean under M0), walking like M0 costs about 1.4 in D + O and gains
  about 0.73 in T. For most of the simplex, standing is the better policy
  under the objective as written. A gentler gait than M0's might score
  better. This was not measured.

The V4-C, V4-C2 and V4-C3 semantic analyses were all made on policy
families that mostly stand. Their registered results stand as recorded. How
they are read changes: "raising w_D does not improve D" is, at least in V4-C3,
"a standing policy has no D left to improve".

## What this does not show

- Whether V4-C policies stand. Their training T reward is at the standing
  level, but this protocol was run only on V4-C3.
- Whether adding a preference-invariant substrate reward would produce
  walking. T⁺ partly stepping suggests gait discovery is possible where T
  dominates, and that the problem is the objective balance at most weights.
  It could equally be missing gait shaping. Separating the two needs its
  own contract.
