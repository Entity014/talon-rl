# B1-P2 — Adaptive-KL Learning-Rate Control

Status: **CLOSED — FAIL / PARTIAL EFFECT**

B1-P1 is **CLOSED — FAIL / partial mechanism effect**. It triggered actor-only KL early stopping frequently and reduced some late drift, but all three update-500 deterministic gates failed. P2 replaces P1; it must not combine with P1 early stopping.

## Hypothesis

Continuous KL-controlled step-size adaptation may preserve deterministic capability better than terminating actor epochs after a large update. The only changed formulation element is the PPO optimizer learning-rate schedule.

## Inherited B0.1 contract

Reward, environment, architecture, fixed `vx=0.5`, scheduled std `0.82→0.10`, 4096 environments, 500 updates, seeds 0/1/2, scalar rollout, and the deterministic 64-state × 500-step monitor/gate remain unchanged.

## P2 mechanism

At the end of each full PPO actor epoch, compute the analytic pre-tanh Gaussian KL on the entire rollout observation batch. PPO continues normally; there is no early stop, rollback, adaptive penalty, or P1 logic.

P2 uses a **co_rl-inspired adaptive-KL learning-rate rule**, while preserving actor/critic separation for causal isolation. It is not intended to be a byte-identical reproduction of co_rl PPO. In particular, co_rl checks KL inside its minibatch loop and changes one optimizer holding all actor-critic parameters; P2 intentionally uses the thesis-specific separation below.

Proposed P2 contract:

- `desired_kl = 0.01`;
- if KL `> 0.02`, actor optimizer learning rate is divided by `1.5`;
- if KL `< 0.005`, actor optimizer learning rate is multiplied by `1.5`;
- clamp actor LR to `[1e-5, 1e-2]`;
- critic LR/schedule remains unchanged;
- evaluate at the declared full-epoch boundary only, with no threshold sweep.

Implementation uses separate actor and critic optimizer parameter groups so that only the actor LR changes; the critic update remains on its predeclared LR and schedule.

The exact KL estimator direction and full-rollout evaluation semantics must match B1-P1's analytic estimator. `approx_kl` remains audit logging only. The LR change observed after epoch `e` takes effect beginning with epoch `e+1`; it never retroactively changes the completed step. If `e` is the final epoch of an update, the changed LR is used starting in the next update.

The current actor LR is optimizer state and must be serialized in every checkpoint and restored exactly on resume. It must not be re-derived from KL history.

## Required logging

Every update records analytic KL, sampled `approx_kl`, actor LR before/after, LR up/down event, ratio mean, clip fraction, actor gradient norm, policy/value loss, scheduled std, and update index. Checkpoint/resume must preserve the adaptive LR state and update index.

## Design-freeze gate

Before training, freeze the equations, thresholds, LR bounds, update boundary, actor/critic optimizer scope, logging schema, hashes, and B0 monitor contract. Run unit tests for LR increase/decrease/clamping, no P1 early-stop fields, critic continuation, std invariance, checkpoint/resume, and B0 flag-off equivalence. Then run a 4–16 environment lifecycle smoke with normal and controlled KL cases.

## Experiment-completion gate

After freeze, run seeds 0/1/2 at the B0.1 budget. Decide only from update-500 deterministic gates: survival ≥90%, velocity MAE ≤0.15 m/s, tilt p95 ≤15°, max tilt ≤30°, and no numerical failures. If P2 fails but clearly reduces abrupt drift, record it as partial and consider a stability-margin formulation; do not immediately stack P1+P2 or sweep thresholds.
