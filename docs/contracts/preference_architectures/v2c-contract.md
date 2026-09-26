# V2-C Contract — Preference-Gated Modular Residual Escalation

Status: PREDECLARED BEFORE V2-C TRAINING

## Frozen

- Foundation V2 in full
- V2-B direct preference path
- V2-A learned preference embedding
- V2-B single-site FiLM path
- seed 73001 for training screens
- 75-update budget unless a short authority screen is explicitly predeclared
- expanded current-policy critic support
- critic body / objective-specific linear heads / ridge lambda = 1
- GAE lambda = 0.95
- repaired PPO/action-logprob semantics
- objective definitions and normalization
- semantic evaluator, matched reset suites, seeds, and thresholds

## Treatment only

Add preference-gated private residual capacity after the V2-B post-FiLM hidden feature and before the existing embedding/action readout.

Architecture:

    h_film = V2-B post-FiLM hidden feature
    g(w) = softmax(router(w))              # 4 experts
    r_i = expert_i(h_film)                 # small bottleneck residuals
    h_mod = h_film + sum_i g_i(w) r_i
    actor_features = concat(h_mod, embedding(w))

The router is deterministic and preference-only.

Each expert is a 128 -> 32 -> 128 residual block. The final 32 -> 128 layer is zero-initialized, so every expert output is exactly zero at V2-C0 initialization.

Router weights and bias are zero-initialized, so routing is exactly uniform at initialization.

Therefore:

    V2-C(initial) == V2-B(initial)

for actor mean, stochastic distribution, transformed action/log-probability, critic/value path, and environment action.

## Causal question

Does partial preference-gated parameter isolation allow semantic competencies to accumulate and persist when increasing conditioning expressivity on a mostly shared actor did not?

This is a parameter-isolation test, not an adaptive-hyperparameter or retention intervention.

## Ladder

V2-C0 — function-preserving implementation gate; no training

V2-C1 — modular authority / specialization screen
- 75 updates, seed 73001, same Foundation-V2 support/critic protocol as V2-B1
- heavy-preference routing pairwise L2 mean >= 0.02
- maximum router deviation from uniform >= 0.02
- expert residual max norm > 1e-3
- mean pairwise expert-output diversity > 1e-3
- expert gradient-share pairwise L2 mean >= 0.02 across T/A/O/S-heavy probes
- router gradient observed (>1e-7 at least once during training)
- modular action authority > 1e-4
- masking the complete modular path must reduce either pairwise action separation or preference-Jacobian norm by > 1e-4
- single-expert mask effects must vary by preference (mean per-expert preference std > 1e-5)
- early and late critic EV remain positive with negative fraction <= 0.25
- combined critic negative fraction <= 0.25
- PPO ratio max error <= 1e-4
- last-10-update termination fraction < 0.5

Router entropy, the full 4x4 routing matrix, expert residual-norm matrix, expert-gradient-share matrix, and single-expert masking matrix are reported diagnostically but no expert identity is preassigned to any objective.

V2-C2 — exact frozen semantic gate
- same endpoint suites
- same continuum suites
- same thresholds
- primary question: do semantic successes accumulate rather than rotate?

V2-C3 — retention/path audit only if V2-C2 shows meaningful semantic improvement

V2-C4 — parameter-matched dense residual control only if V2-C3 supports a retention gain

Three-seed confirmation and deployment metrics are authorized only after the preceding gates pass.

## Stop rule

Stop the V2-C branch immediately if:
- V2-C0 is not exactly function-preserving,
- V2-C1 fails to establish modular authority/specialization, or
- V2-C2 reproduces the same winner-rotation / incomplete semantic pattern without meaningful improvement.

No expert-count, router-temperature, depth, learning-rate, lambda, entropy, objective, critic, support, or threshold tuning is authorized inside this branch.
