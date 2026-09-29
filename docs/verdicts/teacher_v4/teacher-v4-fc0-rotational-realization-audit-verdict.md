# Teacher V4 — FC-0 Rotational-Stability Realization Audit Verdict

Status: **FROZEN. Registered result (primary, FB-2a): F_rate (the current R), F_osc and F_acc are all "candidate". None is task-confounded or O-redundant, and all are informative at matched task. FC-0 therefore finds no measurement defect in `ang_vel_xy_l2` that separates it from the alternatives. Descriptively, the R failure is one of authority: under R⁺ every rotational feature gets worse, not better (R⁺ vs C: F_rate +3 % to +49 %, F_osc +8 % to +19 %, posture +6 % to +26 %). The R preference does not steer rotation at all while the task is enforced.**
Date: 2026-09-29
Contract: [teacher-v4-fc0-rotational-realization-audit-contract.md](../../contracts/teacher_v4/teacher-v4-fc0-rotational-realization-audit-contract.md) (r2, frozen with `fc0_audit.py` at `3614253`, before any trace)
Output: `runs/teacher_v4_fc-2026-09-29/fc0/fc0_audit.json`, traces in `traces/`

Primary bank: 14 FB-2a behaviors with replay tl ≥ 0.40 (79101 R vertex
excluded). Secondary: FB-2, F4, V4-C s73102 (descriptive).

## Registered classification (primary)

| feature | class | η² behavior, tl-matched | η² after task residual | η² O-conditioned | η² policy (diag.) | median \|ρ\| v_cmd / speed / cadence / F_O |
|---|---|---|---|---|---|---|
| F_rate (current R) | candidate | 0.45 | 0.45 | 0.38 | 0.23 | 0.18 / 0.44 / 0.69 / 0.12 |
| F_osc | candidate | 0.37 | 0.35 | 0.18 | 0.08 | 0.11 / 0.09 / 0.43 / 0.62 |
| F_acc | candidate | 0.43 | 0.42 | 0.38 | 0.14 | 0.11 / 0.32 / 0.61 / 0.27 |

Secondary (descriptive, cannot flip the primary class): all three are
candidates. F_rate and F_acc carry the task-association flag (speed ρ 0.61
and 0.55). F_osc has no flag.

## Descriptive

- **R⁺ worsens rotation.** R⁺ vs C (F_rate / F_osc / F_O):
  79101 +13 % / +8 % / +6 %; 79102 +49 % / +18 % / +17 %;
  79103 +3 % / +19 % / +26 %. The only "improves while posture worsens"
  flag is F_acc in 79103 (−2 %).
- **The R vertex tilts.** F_O is 0.068–0.093 rad (4–5°) against 0.013–0.014
  at C, with more oscillation (F_osc 0.017–0.023 vs 0.009–0.011).
- **The O vertex rotates fastest.** F_rate is 0.19–0.96 against 0.07–0.42 at
  C, with high cadence. The O preference buys low tilt with more roll/pitch
  rate: R and O trade off inside locomotion.
- F_rate is strongly tied to cadence (median ρ 0.69). F_osc is the least
  tied to speed and cadence (0.09 / 0.43) and the least policy-dependent
  (η² policy 0.08), but it is the most associated with posture (0.62).

## Reading

- FC-0 does not support replacing `ang_vel_xy_l2` on measurement grounds.
  On FB-2a it is informative at matched task, and it is neither
  task-confounded nor O-redundant. F_osc is a cleaner-looking alternative
  (the least gait-coupled) but not better by the frozen criteria.
- The FB-2a failure is therefore not "R measures the wrong thing". **The R
  preference does not move the policy toward lower rotation.** R⁺ makes
  every rotational measure worse, and the R vertex trades rotation for
  tilt. This points to preference **authority** for R (credit or advantage
  share, or the R–O interaction), not to its realization.

**Limits.** Selection data only (no candidate validated). One
calibrated-task training round (FB-2a, 3 policies). The O-conditioned test
bins posture, which is itself moved by the preferences.
