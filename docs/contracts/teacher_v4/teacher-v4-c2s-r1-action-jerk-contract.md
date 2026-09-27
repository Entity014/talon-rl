# Teacher V4 — V4-C2S-R1 Train-on-S1 Causal Test Contract

Status: **PREDECLARED — FROZEN 2026-09-27 before any S1 training**
Branch: `v4-c2-semantic-preservation`

## Question

V4-C2S found every smoothness candidate nearly redundant with A, but on
policies trained with S0. When a policy is actually trained with S1 (action
jerk), does A–S still collapse into one behavioral dimension?

One candidate only, chosen before training: **S1 = ‖a_t − 2a_{t−1} + a_{t−2}‖²**,
the reward term `action_jerk_l2`, weight −0.01.

## Control and treatment (paired by seed and fold)

- Control: the six V4-C G1 runs (G1-2 and G1-3 × seeds 73101–73103), trained
  with S0.
- Treatment: the same folds, seeds, architecture, T/A/O definitions,
  env (`Isaac-Talon-A1-V4C-S1-v0` = V4-C env + the action-jerk term; physics
  identical, divisors T/A/O reproduced bit for bit), objective-set sampler,
  PPO recipe, 29.49M-sample budget, and iteration-300 checkpoint. The only
  change: the S objective is `action_jerk_l2` with divisor
  **S1_DIVISOR = 0.21491182** (T3-B protocol on the M0 policy, measured
  before this contract). `train_v4c.py --s-objective action_jerk`.
- Declared difference: the control runs used the float32 `RunningMeanStd`
  (pre-`61a483d`); the treatment runs use the fixed float64 version.
- Term verified against a hand computation (max error 3e-8).

## Re-audit

`s_candidates_audit.py` on both groups (12 runs), the same protocol as V4-C2S,
which records S_T, S_A, S_O, S0–S3 per step. Analysis follows V4-C2F with the
S column = S1 for the treatment (S-heavy = S1-heavy there) and S = S1 also
computed on the control, for pairing:

- layers 1–2: |ρ| per step and per 32-step window, off-diagonal mass, PC1
  (snapshots), for every pair among T, A, O, S1;
- layer 3 (treatment only, where S-heavy means S1-heavy): CI signs of
  SS − SA and AA − AS over the 6 runs.

## Outcome rule (fixed now; treatment group)

- **Separates:** A–S1 is not the most redundant pair on any of |ρ| per step,
  off-diagonal mass and PC1; |ρ(S1, T)| and |ρ(S1, O)| ≤ 0.16 per step and
  ≤ 0.23 per window (the C2S eligibility bound); and SS − SA is positive in
  ≥ 4/6 runs. → Keep T/A/O/S1; semantic-specificity training next.
- **Still redundant:** A–S1 is the most redundant pair on all three
  measures, and SS − SA is positive in ≤ 3/6 runs. → The smoothness family is
  subsumed by the gait's stability dynamics; merging A and S is justified.
- **Mixed:** anything else. Characterize with more seeds first; no merge, and
  no choice of S from the best seed.

Paired change ρ(A, S1)_treatment − ρ(A, S1)_control per seed is reported
descriptively, as are training health and the T/A/O m = 4 endpoints.
