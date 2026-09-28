# Teacher V4 — V4-C3 Substrate Attribution Contract (H2, read-only)

Status: **FROZEN 2026-09-28 before any measurement.** Diagnostic only. It trains nothing, changes no gate and does not revise the V4-C3 verdict.
Branch: `v4-c2-semantic-preservation`
Follows: [V4-C3 verdict](../../verdicts/teacher_v4/teacher-v4-c3-verdict.md)

## Question

V4 training rewards only T, D (= A) and O. The stock non-objective terms
(`lin_vel_z_l2`, `dof_torques_l2`, `dof_acc_l2`, `feet_air_time`) are computed
but have zero columns in the training reward. M0 was trained on all of them.

**H2 (missing shared locomotion substrate):** D fails in V4-C3 because the
policy family has left the locomotion manifold that M0's non-objective terms
kept it on. Raising w_D moves the policy further off that manifold, not
toward better D.

Competing, not tested here: H1 D realization itself, H3 critic / return
representation, H4 combination.

## Protocol

Same switch-controlled branch protocol as `c3_semantics.py` (anchor group):
512 center-warm-up snapshots (env seeds 0, 1 × 256 envs), a discarded
burn-in, then five 128-step branches restored in place with cyclic position
balance: **C, T⁺, A⁺, O⁺, M0**. Heavy = 0.70 / 0.15 / 0.15. M0 =
`runs/m0_1_seed0_2026-09-22/model_299.pt` in `V1CSharedActorCritic` with
w = [1, 0, 0] (function-preserving), deterministic mean, clamped to ±1.
Every V4 branch is deterministic (`act_inference`). Steady window 33–128.

Checkpoints: the six V4-C3 runs at iteration 300.

Logged per step: every active reward term (weighted, as in the reward
manager), S_T / S_A / S_O, ‖ω_xy‖, tilt, v_z, four foot-contact flags,
‖q̈‖, ‖τ‖.

**Substrate terms:** `lin_vel_z_l2`, `dof_torques_l2`, `dof_acc_l2`,
`feet_air_time`. All are higher-is-better as weighted.

Note: the contact sensor's air-time state is not part of the snapshot, so
`feet_air_time` is only reliable after each foot's first touchdown in a
branch. The steady window starts at step 33, after that.

## Measures

Per run, per substrate term k, steady-window means over snapshots:

- **Level:** L_k = r_k(C) − r_k(M0). Negative = the V4 center is worse than M0.
- **Response:** Δ_k(j) = r_k(j⁺) − r_k(C) for j ∈ {T, A, O}.

Two-sided 95% simultaneous bootstrap bounds (snapshot = unit, max-t over the
4 terms × 4 contrasts). An effect is **material** if its bound excludes 0
**and** |effect| ≥ 0.10 × |r_k(M0)|.

## Pre-declared reading

- **S1 (off-manifold level):** in a run, at least 2 of 4 substrate terms have
  a material negative L_k.
- **S2 (D preference degrades substrate):** in a run, at least 1 substrate
  term has a material negative Δ_k(A).

| outcome | reading |
|---|---|
| S1 and S2 in ≥ 4/6 runs | **H2 supported.** Justifies a preference-invariant substrate reward screen. |
| S1 in < 3/6 runs | **H2 not supported.** The V4 center is on M0's manifold, so a missing prior cannot explain D. |
| otherwise | **Mixed.** Report as is. No intervention justified by this diagnostic alone. |

Fidelity is not re-measured here. The V4-C3 fidelity-invalid runs (G1-1
s73101, G1-2 s73103) are included and flagged, and the reading is also
reported without them.

## Descriptive (not gated)

- The full Δ matrix over all logged terms and S_T / S_A / S_O, for T⁺, A⁺,
  O⁺ and M0 (vs C).
- Gait structure per condition: foot-contact-count distribution, diagonal
  pair (trot) fraction, touchdown rate.
- Phase-conditioned ‖ω_xy‖²: mean at touchdown steps vs other steps, per
  condition. It asks whether A penalizes gait-intrinsic angular motion.
- Traces of the first 64 envs saved for later temporal analysis.

## If H2 is supported

The next step is a screened intervention, not a D redefinition. Substrate
terms enter as a preference-invariant reward
r = Σ_i w_i R_i + R_shared, never inside D. Order: lin_vel_z → torque/acc →
feet_air_time → full stock. That needs its own contract.
