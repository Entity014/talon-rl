# V2-K0 Function-Preserving Gate Verdict

Status: **FROZEN — V2-K0 PASS; V2-K1 AUTHORIZED**

Date: 2026-09-24

## Treatment

V2-K combines two previously separate ideas:

- V2-C private residual capacity;
- V2-H continuous preference-conditioned coefficients.

Architecture:

    h' = h + sum_k c_k(w) E_k(h)

where the four private modules are unlabeled and the coefficient mapping has no
softmax constraint.

At initialization:
- private module parameters and outputs are nonzero;
- coefficient output is exactly zero;
- combined residual is exactly zero.

Therefore:

> V2-K(initial) ≡ V2-B(initial)

while the coefficient mapping can receive gradients immediately.

## Gate result

All K0 invariants pass.

Exact identity vs V2-B:
- deterministic action max difference = **0.0**
- pre-tanh mean difference = **0.0**
- critic difference = **0.0**
- same-latent log-prob difference = **0.0**
- PPO ratio error = **0.0**
- environment raw-action difference = **0.0**
- all pre-existing V2-B parameter difference = **0.0**
- 64-step paired no-update action difference = **0.0**
- paired-smoke termination events = **0**

Coefficient / private-module initialization:
- coefficient max absolute value = **0.0**
- combined private residual max absolute value = **0.0**
- private-module mean output norms = **[0.0978, 0.0997, 0.0821, 0.0920]**
- mean pairwise private-module output diversity = **0.1320**
- minimum pairwise diversity = **0.1215**

Checkpoint roundtrip and normalized four-objective contract also pass exactly.

## Decision

**V2-K0 PASS.**

V2-K1 private-subspace authority/diversity screen is authorized.

Primary artifacts:
- `talon_rl/v2k_actor_critic.py`
- `scripts/rl/v2k0_function_preserving_gate.py`
- `runs/v2k0_function_preserving_gate-2026-09-24/v2k0_report.json`
- `runs/v2k0_function_preserving_gate-2026-09-24/v2k0_init.pt`
- `docs/contracts/preference_architectures/v2k-contract.md`
