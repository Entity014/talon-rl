# B1-S1 — ACAPS-Style Stability-Margin Regularization

Status: **CLOSED — FAIL / PARTIAL EFFECT**

B1-P2 is **CLOSED — FAIL / PARTIAL EFFECT**: adaptive KL learning-rate control preserved survival and velocity tracking in seeds 0/1, but tilt remained outside the gate and seed2 failed strongly. The remaining hypothesis is insufficient local stability margin rather than policy-update drift alone.

## Formulation choice

B1-S1 is a declared composition: **B1-P2 adaptive-KL actor LR + ACAPS-style actor regularization**. This is not a single-variable intervention from B0.1. P2 is retained because its preservation effect is already established; ACAPS addresses the residual local robustness bottleneck. P1 early stopping remains disabled. The term ACAPS-style means an ACAPS-inspired formulation, not a reproduction of the `SRMPPO` source: S1 does not use return-derived `alpha_t`, uses mean-to-mean temporal consistency, and uses the explicitly specified perturbation below.

## Inherited contract

Keep B0 reward, terminal semantics, fixed `vx=0.5`, architecture, scheduled std `0.82→0.10`, 4096 environments, 500 updates, seeds 0/1/2, and the 64-state × 500-step deterministic monitor/gate unchanged. No MOPPO/preference/vector normalization state is introduced.

## Proposed regularization

Let `mu(o)` be the actor pre-tanh mean.

### Spatial consistency

For each rollout observation `o`, draw one independent normalized policy-observation perturbation `δ ~ Normal(0, 0.02² I)`, clip each component to `[-0.05, 0.05]`, detach `δ`, and apply an explicit mask that excludes command dimensions `42:45` (`vx, vy, omega_z`). Add:

`L_spatial = mean(||mu(o + δ) - mu(o)||²)`.

The perturbation is applied only to policy observations and is not stepped through the simulator. The monitor states remain evaluation-only.

### Temporal smoothness

Construct consecutive pairs from time-major rollout storage before any PPO minibatch permutation. For consecutive rollout observations `(o_t, o_{t+1})` from the same lane, add:

`L_temporal = mean(||mu(o_{t+1}) - mu(o_t)||²)`.

Pairs crossing a done/reset boundary are masked out using the stored same-episode mask. No artificial temporal pairs are constructed across episode resets.

### Proposed coefficients and scope

The design proposal to freeze is `lambda_spatial = 0.10` and `lambda_temporal = 0.05`. Both terms are added to the **actor loss only**; the critic loss and critic optimizer are unchanged. P2's actor LR adaptation uses analytic KL exactly as frozen in B1-P2, and the regularizers do not alter scheduled std.

These coefficients are not to be swept during the first causal test. If they are rejected at design review, B1-S1 remains un-frozen rather than silently tuning them during training.

Before freeze, run a read-only rollout scale audit reporting unweighted PPO actor loss, each regularizer, weighted contributions, and separate gradient norms. The purpose is only to reject pathological scale (dominant or numerically dead regularization), not to tune coefficients.

## Required logging

Every update records spatial loss, temporal loss, each weighted contribution, actor LR/KL events from P2, critic LR, scheduled std, reset-pair count, perturbation norm statistics, gradient norm, and update index. Checkpoint/resume must preserve P2 optimizer state and all S1 configuration fields.

## Design-freeze gate

Before training, lock the equations, perturbation distribution/clipping, coefficients, done-mask semantics, actor-only scope, P2 inheritance, hashes, logging schema, and unchanged deterministic gate. Unit tests must verify perturbation reproducibility, reset masking, finite losses, actor-only gradients, critic-LR invariance, P2 LR adaptation, std invariance, checkpoint/resume, and S1-off equivalence.

## Experiment-completion gate

Run the same three seeds and budget only after freeze. Decide from update-500 monitor results using the unchanged gate: survival ≥90%, velocity MAE ≤0.15 m/s, tilt p95 ≤15°, max tilt ≤30°, and no numerical failures. Report separately whether S1 improves tilt/stability and whether P2 preservation remains intact; do not change reward/std or stack another optimizer intervention mid-run.
