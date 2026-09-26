# V2-K Contract — Hyper-Conditioned Private Residual Combination Rescue

Status: PREDECLARED BEFORE V2-K TRAINING

## Purpose

V2-K is a bounded **combination rescue branch** motivated by complementary failures of V2-C and V2-H.

V2-C:
- private residual capacity became active;
- learned soft routing remained near-uniform;
- preference-dependent allocation did not emerge.

V2-H:
- preference-conditioned coefficients and local parameter variation emerged;
- generated policy deltas remained dominated by a common absolute direction.

V2-K therefore combines:
- genuinely private residual modules;
- continuous preference-conditioned coefficients without softmax routing.

Architecture:

    h = V2-B post-FiLM hidden
    c(w) = coefficient network
    E_k(h) = private residual module k

    h' = h + sum_k c_k(w) E_k(h)

The modules are unlabeled and are not assigned to Tracking, Angular,
Orientation, or Smoothness.

## Frozen foundation

- Foundation V2
- complete V2-B direct preference path
- V2-A preference embedding
- V2-B single-site FiLM path
- critic architecture / support / freshness protocol
- GAE lambda = 0.95
- repaired PPO action/log-probability semantics
- objective definitions and normalization
- semantic evaluator, seeds, suites, and thresholds
- default V2-B authority-screen training budget and seed

## Treatment only

Add:
- four private residual modules, each 128 -> 32 -> 128
- coefficient network 4 -> 16 -> 4
- unconstrained deterministic coefficients, no softmax normalization

At K0:
- private module parameters are nonzero;
- coefficient output layer is exactly zero;
- therefore c(w)=0 and combined private residual=0 for all w.

Thus:

    V2-K(initial) == V2-B(initial)

while coefficient gradients can be nonzero immediately.

## K0 — function-preserving gate

No training.

Must pass:
- deterministic action identity vs V2-B
- pre-tanh mean identity
- same-latent transformed log-probability identity
- critic/value identity
- environment raw-action identity
- PPO ratio invariant
- all V2-B tensors copied exactly
- coefficient output exactly zero on standard preference probes
- combined private residual exactly zero
- private module outputs nonzero and mutually diverse
- checkpoint roundtrip exact
- 64-step paired no-update smoke identical to V2-B
- normalized four-objective contract unchanged

## K1 — private-subspace authority / diversity

Authorized only if K0 passes.

K1 asks only:

> Can continuous preference-conditioned coefficients drive genuinely distinct
> private residual subspaces without the common-direction collapse observed in
> V2-C and V2-H?

Use 75 updates, seed 73001, and the same Foundation-V2 support/critic protocol.

Predeclared PASS criteria:

### Preference -> coefficient authority
- heavy-preference mean pairwise coefficient distance >= 0.02
- centered heavy-preference coefficient s2/s1 >= 0.05
- coefficient Jacobian Frobenius norm > 1e-3

### Preference -> private contribution diversity
For each heavy preference, define the mean private contribution

    r(w) = sum_k c_k(w) E_k(h)

on a fixed-state probe set.

Require:
- mean pairwise residual-vector distance >= 0.01
- mean pairwise (1 - cosine) of absolute residual vectors >= 0.02
- centered residual manifold s2/s1 >= 0.05
- at least two modules have nontrivial contribution share (> 0.10) for at
  least one heavy preference
- mean pairwise L2 distance between module-contribution-share vectors >= 0.02

### Private modules -> action authority
- mean ||a_full - a_private_masked|| > 1e-4
- masking private path reduces either pairwise action separation or
  preference-Jacobian norm by > 1e-4
- at least one single-module mask has preference-specific action effect
  with mean per-module preference std > 1e-5

### Trainability / non-collapse
- coefficient output-layer gradient observed > 1e-7
- private-module gradient observed > 1e-7 after coefficients become nonzero
- private module outputs remain nonzero and mutually diverse

### Foundation guardrails
- early and late critic EV > 0
- early and late critic negative fraction <= 0.25
- combined critic negative fraction <= 0.25
- PPO ratio max error <= 1e-4
- last-10-update termination fraction < 0.5

Reported diagnostics:
- T/A/O/S/C coefficient matrix
- coefficient covariance / singular spectrum
- private residual cosine matrix
- residual centered singular spectrum
- module contribution-share matrix
- single-module masking matrix
- full vs private-masked action separation
- full vs private-masked action-preference Jacobian
- coefficient and module gradient norms

No semantic endpoint judgment is allowed in K1.

## K2 — semantic accumulation

Authorized only if K1 passes every predeclared authority/diversity gate.

Use the exact frozen endpoint and continuum semantic contract.

Primary question:

> Do semantic successes accumulate across objectives rather than rotate?

## Stop rule

Stop immediately if:
- K0 is not exact,
- K1 private residuals remain dominated by a common direction,
- coefficient/module allocation does not differ meaningfully by preference,
- private-path masking does not reduce preference authority, or
- Foundation V2 fails.

No tuning of module count, bottleneck size, coefficient-network depth,
coefficient normalization, learning rate, lambda, entropy, objectives, critic,
support protocol, or semantic thresholds is authorized inside this branch.
