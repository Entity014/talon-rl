# B1 — Deterministic Policy Preservation / Stability-Margin Design

Status: **DESIGN DRAFT — B1-P1 SELECTED FOR DESIGN; NOT FROZEN**

Date: 2026-09-21

## Scope

B1 addresses the failure mode exposed by B0.1: deterministic locomotion capability can appear transiently and then disappear before the frozen final checkpoint. B0.1 remains the baseline; its reward, environment, fixed command, scheduled standard deviation, 4096-env budget, and 500-update protocol are not changed by this document.

The design question is:

> How can deterministic capability that appears during training be preserved, or made robust to small policy/state perturbations, without changing the B0 reward semantics prematurely?

This document defines competing formulation branches. It does not select one yet.

## Evidence from B0.1

- Seed 1: update 25→50 had survival `1.00→0.00`; same-state action mean ΔL2 was approximately `1.64`, with actor-output cosine `≈0.822`. This is an abrupt policy-drift signature.
- Seed 0: update 450→475 had survival `1.00→0.359`; same-state action mean ΔL2 was approximately `0.79`, with actor-output cosine `≈0.996`. This is compatible with a narrow stability margin or state-distribution sensitivity.
- Seed 2: same-state action drift was approximately `0.38` and the full deterministic gate was never reached. Drift alone is therefore not sufficient as a universal explanation.
- Existing artifacts do not contain replayable near-boundary simulator states or per-step trajectories. No stronger causal claim is made here.

## Candidate branches

### B1-P — Policy preservation

Constrain update-to-update deterministic mean-policy movement, using one explicitly specified mechanism:

- frozen-state mean anchor, or
- deterministic-policy KL / trust-region penalty, or
- an equivalent bounded mean-action displacement constraint.

The mechanism must be evaluated on the same 64 frozen states used by the monitor and must not silently constrain stochastic sampling behavior differently from the stated contract.

### B1-S — Stability margin

Train robustness around states where the deterministic policy remains viable. The state-neighborhood construction, perturbation distribution, and weighting must be specified before training. Candidate perturbations may include small proprioceptive/state disturbances and short local rollout neighborhoods, but they must not introduce an unbounded new objective or hidden preference state.

## Shared non-goals and guardrails

- Do not change B0 reward semantics, terminal semantics, fixed command, or scheduled-std formula as part of B1 diagnosis.
- Do not cherry-pick the best transient checkpoint as a replacement for the frozen B0.1 final checkpoint.
- Do not add preference/MOPPO state, vector-reward normalization, D3PO/diversity state, or `log_std` annealing.
- The deterministic monitor remains `tanh(actor_mean)` on 64 frozen reset states × 500 steps.
- Every B1 run must record actor-output drift, parameter displacement, policy loss, value loss, and all monitor artifacts. If KL, ratios, clip fraction, gradient norm, returns, advantages, or value error are required, they must be added to the runner before the design is frozen.

## Branch-selection protocol

Select one branch from the current evidence before starting full training. Do not run both branches as a six-run screening experiment. B1-P is the leading candidate because B0.1 directly shows abrupt deterministic-policy drift, while B1-S requires a new state-neighborhood construction that is not supported by replayable B0.1 boundary states. The selected first mechanism is B1-P1, target-KL early stopping; its exact design is in `b1-p1-target-kl-early-stopping-freeze-draft.md`.

Before design freeze, run only read-only pre-design calculations that use existing artifacts (including consecutive-policy KL). Then select exactly one preservation mechanism and freeze it. B1-S remains the fallback if the selected B1-P mechanism is not justified or does not address the predeclared gate.

## Design-freeze gate

Training is authorized only after the selected B1 branch has:

- an explicit mathematical objective and coefficient/configuration manifest;
- no hidden B0.1 formulation changes;
- deterministic monitor reproducibility and training-state isolation;
- a predeclared acceptance rule for survival, velocity MAE, tilt, and numerical health;
- an explicit decision about whether the branch targets abrupt policy drift, narrow stability margin, or both;
- code/config/manifest hashes for the selected formulation and instrumentation;
- an independent monitor-state policy (the 64 evaluation states must not become the training anchor set);
- implementation sanity tests for estimator correctness, actor-only stopping, critic continuation, std invariance, resume, and B0.1 flag-off equivalence.

## Experiment-completion gate

After design freeze and training, issue a verdict only when the same three-seed protocol has complete update-500 checkpoints and artifacts, no hash/config mismatch, and the predeclared deterministic acceptance rule has been evaluated. Update-250 evidence may be recorded for review but cannot replace the update-500 decision checkpoint.

Until the design-freeze gate passes, B0.1 remains **CLOSED — FAIL**, and B1 remains **DESIGN DRAFT — NOT FROZEN**.
