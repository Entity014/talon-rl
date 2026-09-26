# V2-H Contract — Preference-Conditioned Policy-Parameter Manifold

Status: PREDECLARED BEFORE V2-H TRAINING

## Motivation

The final bounded branch tests whether winner rotation is caused in part by forcing all preferences to share one policy-parameter realization.

V2-H changes the conditional-policy parameterization:

    w -> coefficient network c(w)
    W(w) = W0 + sum_k c_k(w) U_k V_k

Only the actor action-head weight receives a preference-conditioned low-rank delta.

The bases are latent and carry no objective labels.

## Frozen

- Foundation V2
- V2-B direct preference path
- V2-A learned preference embedding
- V2-B single-site FiLM path
- critic architecture and support/freshness protocol
- GAE lambda = 0.95
- repaired PPO/action-logprob semantics
- objective definitions and normalization
- semantic evaluator, matched-reset seeds/suites, thresholds
- training budget and default training seed used by the V2-B authority screen

## Treatment only

Add:
- K = 4 latent low-rank weight bases
- rank = 4 per basis
- a small deterministic coefficient network 4 -> 16 -> 4
- preference-conditioned delta only on actor_mean.weight

Generated effective action-head weight:

    W(w) = W0 + sum_k c_k(w) B_k
    B_k = U_k V_k

The final coefficient layer is zero-initialized, while U_k and V_k are nonzero.

Therefore at H0:

    c(w) = 0
    delta W(w) = 0
    V2-H(initial) == V2-B(initial)

exactly for every preference.

## H0 — function-preserving gate

No training is allowed.

Must pass:
- deterministic action identity vs V2-B
- pre-tanh mean identity
- same-latent transformed log-probability identity
- critic/value identity
- environment raw-action identity
- PPO ratio invariant
- all V2-B tensors copied exactly
- coefficient output exactly zero on standard preference probes
- generated weight delta exactly zero
- low-rank bases nonzero
- checkpoint roundtrip exact
- 64-step paired no-update smoke identical to V2-B
- normalized four-objective contract unchanged

## H1 — parameter-manifold authority

Authorized only if H0 passes.

H1 uses 75 updates, seed 73001, and the same Foundation-V2 support/critic protocol as V2-B1. It asks only whether preference generates distinct policy parameters and whether those parameters have causal action authority.

Predeclared PASS criteria:
- mean pairwise coefficient distance across T/A/O/S-heavy preferences >= 0.02
- mean pairwise generated-weight-delta Frobenius distance >= 1e-4
- generated-delta direction diversity: mean pairwise (1 - cosine) across heavy preferences >= 0.02
- centered heavy-preference coefficient matrix has at least two meaningful directions, with s2/s1 >= 0.05
- centered heavy-preference generated-delta matrix has at least two meaningful directions, with s2/s1 >= 0.05
- local parameter-manifold Jacobian at center preference has at least two meaningful directions, with s2/s1 >= 0.05
- coefficient Jacobian Frobenius norm > 1e-3
- generated-parameter Jacobian Frobenius norm > 1e-4
- coefficient-network output-layer gradient observed (>1e-7 at least once)
- low-rank basis gradient observed (>1e-7 at least once after coefficients become nonzero)
- mean hyper-path action authority ||a_full-a_masked|| > 1e-4
- masking generated delta reduces either pairwise action separation or preference-Jacobian norm by > 1e-4
- early and late critic EV remain positive with negative fraction <= 0.25
- combined critic negative fraction <= 0.25
- PPO ratio max error <= 1e-4
- last-10-update termination fraction < 0.5

Reported diagnostics additionally include:
- T/A/O/S/C coefficient matrix
- coefficient covariance / singular spectrum
- generated-delta pairwise Frobenius distances and cosine matrix
- basis-contribution shares by preference
- local coefficient and parameter-manifold Jacobian singular values
- full vs hyper-masked action separation and action-preference Jacobian

No semantic endpoint judgment is allowed in H1.

## H2 — semantic accumulation

Authorized only if H1 establishes parameter-manifold authority.

Use the exact frozen endpoint and continuum semantic contract.

Primary question:

> Do semantic competencies accumulate across T/A/O/S instead of rotating among objectives?

## H3 — retention timeline

Authorized only if H2 shows meaningful semantic improvement.

Measure:
- first PASS
- retained-pass fraction
- worst forgetting
- winner-rotation frequency
- semantic competence persistence

## H4 — parameter-matched dense control

Authorized only if H3 supports improved persistence.

Compare against a preference-independent dense residual with approximately matched added parameter count.

This separates:
- gain from extra capacity
from
- gain from preference-conditioned policy-parameter generation.

## Stop rule

Stop immediately if:
- H0 is not exact,
- H1 does not produce distinct parameter realizations with causal action authority, or
- H2 reproduces the prior incomplete/winner-rotation semantic pattern.

No tuning of basis count, rank, coefficient-network depth, learning rate, lambda, entropy, objectives, critic, support protocol, or semantic thresholds is authorized inside this branch.

Generated-weight retention is explicitly out of scope unless H2 first demonstrates semantic accumulation and H3 later identifies residual forgetting.
