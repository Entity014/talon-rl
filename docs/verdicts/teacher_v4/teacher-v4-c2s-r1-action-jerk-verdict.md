# Teacher V4 — V4-C2S-R1 Train-on-S1 Verdict

Status: **FROZEN — STILL REDUNDANT. Trained on action jerk, A–S1 remains the most redundant pair on every measure (ρ 0.80 / 0.90, PC1 0.94), and S1-heavy is worse at S1 than A-heavy in 5/6 runs. Merging A and S is justified.**
Date: 2026-09-28
Branch: `v4-c2-semantic-preservation`
Contract: [teacher-v4-c2s-r1-action-jerk-contract.md](../../contracts/teacher_v4/teacher-v4-c2s-r1-action-jerk-contract.md) (frozen at `42a09ec`, before training)
Runs: `runs/teacher_v4_c2s1_g1_{2,3}_seed{73101,73102,73103}-2026-09-27/` (treatment) and the six V4-C G1 runs (control). Aggregate: `runs/teacher_v4_c2s_r1-2026-09-27/aggregate.json` (`r1_aggregate.py`).

## Training

All six S1 runs completed 300 iterations with no stop gate. Episode length
at iteration 300: 727–989 steps. **One integrity problem:** in G1-2 s73102
the critic collapsed from iteration ≈ 39. EV ≈ 0 on all four objectives to
the end, while the policy kept improving (preference authority 0.66). The
outcome below is the same with and without that run.

## Layers 1–2 (S := S1; medians over 6 runs)

| pair | treatment ρ step / win / off-diag / PC1 | control (S0-trained) ρ step / win / off-diag / PC1 |
|---|---|---|
| TA | −0.11 / −0.12 / 0.54 / 0.62 | −0.01 / 0.04 / 0.51 / 0.63 |
| TO | −0.02 / −0.03 / 0.49 / 0.50 | 0.03 / 0.02 / 0.49 / 0.50 |
| TS | −0.13 / −0.13 / 0.53 / 0.63 | 0.00 / 0.04 / 0.50 / 0.67 |
| AO | 0.24 / 0.35 / 0.41 / 0.60 | 0.09 / 0.15 / 0.49 / 0.61 |
| **AS** | **0.80 / 0.90 / 0.17 / 0.94** | **0.79 / 0.91 / 0.18 / 0.95** |
| OS | 0.14 / 0.24 / 0.45 / 0.53 | 0.00 / 0.02 / 0.54 / 0.53 |

Training on S1 leaves the A–S1 coupling essentially unchanged. Paired per
seed, ρ(A, S1) goes from control to treatment: 0.83→0.82, 0.78→0.74,
0.77→0.83, 0.87→0.79, 0.62→0.80, 0.79→0.64. There is no systematic
reduction.

## Layer 3 (treatment, S1-heavy)

    SS − SA :  neg, neg, neg, zero, neg, neg   (0/6 positive)
    AA − AS :  pos × 6

Asking for S1 does not make actuation smoother than asking for A. Asking for
A makes it at least as smooth, and also improves A.

## Outcome rule (frozen)

A–S1 is the most redundant pair on |ρ| per step, off-diagonal mass and
PC1 (rank 1 on all three), and SS − SA is positive in 0/6 ≤ 3/6 →
**still redundant**. Without the collapsed-critic run: A–S1 ρ 0.80,
off-diagonal 0.17, PC1 0.95; SS − SA 0/5 positive. Same outcome.

## Conclusion

The objection to V4-C2S ("these policies were trained on S0") is answered.
When trained on action jerk, the policy still realizes smoothness and
angular stability as one behavioral dimension, and the S preference has no
specific effect of its own. On this robot, gait and substrate, the
smoothness family tested (action rate, and action jerk as a trained
objective) is subsumed by the gait's stability dynamics. Per the contract,
**merging A and S is justified**.

Side observation: under S1 training, the A–O coupling rose (ρ 0.09→0.24 per
step, 0.15→0.35 per window), as did O–S (0.00→0.14). The objectives' mutual
structure depends on what S is.

## Integrity diagnostic: the collapsed critic (read-only, 2026-09-28)

Offline, on the saved checkpoints; the fixed probe states and nominal e_t;
compared with the healthy G1-2 s73101 S1 run.

- G1-2 s73102 S1: the critic body's **last layer (256 → 128, ELU) is dead**.
  0% of its units are unsaturated on the probes, so c_t is constant (std 0)
  and V varies only with the objective query. The earlier layers are alive
  (91% and 49% unsaturated). The critic-body weights stop changing after
  iteration 50 (norms 20.11 and 14.50 identical at 50, 100 and 300),
  because the saturated ELU passes almost no gradient. Only the query
  head's output bias keeps tracking mean returns. There are no NaN/inf
  values, and Adam state is finite.
- The healthy run has c_t std 0.45–0.65 and 16–30% of units active.
- A survey of every saved checkpoint in all 22 V4-C / V4-C2S-R1 runs finds a
  dead critic only in this run (from ≤ iteration 50).

Classification: **stochastic optimizer collapse (dead-unit saturation of the
last critic layer)**. It is not a numerical fault and not a logging artifact.
A plausible contributor is that the critic's LR follows the actor-KL
adaptive schedule, which reaches 3–5e-3 early. That is not tested and not
changed here.

Integrity gate added to `train_v4c.py` for later rounds: the logged
`critic_feature_std_max` on 1024 rollout states each iteration, and a stop
(the run counts as failed, not replaced) after 5 consecutive iterations with
c_t std < 1e-5.
