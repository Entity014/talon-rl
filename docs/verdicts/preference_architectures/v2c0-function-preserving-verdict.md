# V2-C0 Function-Preserving Gate Verdict

Status: **FROZEN — V2-C0 PASS; V2-C1 AUTHORIZED**

Date: 2026-09-24

## Scope

V2-C0 tests implementation equivalence only. No optimizer step is permitted.

Frozen reference:
- V2-B0 function-preserving initialization
- Foundation V2
- V2-A preference embedding
- V2-B single-site FiLM
- critic/value path
- PPO/action semantics
- objective contract
- environment action path

Treatment only:
- four private residual experts, each 128 -> 32 -> 128
- deterministic preference-only softmax router
- residual insertion after the V2-B post-FiLM hidden feature and before the existing embedding/action readout
- expert output layers initialized to zero
- router logits initialized to zero

Therefore the required initialization contract is:

> V2-C(initial) ≡ V2-B(initial)

## Gate result

All implementation invariants pass exactly.

### Actor / policy identity

- deterministic action max absolute difference vs V2-B: **0.0**
- pre-tanh mean max absolute difference: **0.0**
- same-latent transformed log-probability max absolute difference: **0.0**
- PPO ratio max |ratio - 1|: **0.0**

### Critic / environment identity

- critic value max absolute difference: **0.0**
- environment raw-action max absolute difference: **0.0**
- 64-step paired no-update smoke max action difference vs V2-B: **0.0**
- termination events in paired smoke: **0**

### Modular identity initialization

Router:
- router weight norm: **0.0**
- router bias norm: **0.0**
- route max absolute difference from uniform: **0.0**
- route weights on all standard preference probes: exactly **0.25 each**
- simplex row-sum error: **0.0**

Experts:
- all four expert final-layer weight norms: **0.0**
- all four expert final-layer bias norms: **0.0**
- all expert outputs at the probe state: **0.0**
- routed residual max absolute value: **0.0**

### Treatment isolation

All pre-existing V2-B tensors are copied exactly:
- maximum existing-parameter absolute difference: **0.0**

The only new parameters are:
- preference router weight/bias
- four residual-expert fc1 weight/bias tensors
- four residual-expert zero-output fc2 weight/bias tensors

No critic parameter, existing actor parameter, preference embedding parameter, or FiLM parameter is altered.

### Checkpoint integrity

- action roundtrip difference: **0.0**
- value roundtrip difference: **0.0**
- state hash equality: **PASS**

### Objective contract

The normalized four-objective probe and scalarization contract pass exactly.

## Interpretation

V2-C0 establishes that the modular architecture can be introduced without changing the starting policy, value function, stochastic distribution, transformed action likelihood, or environment action.

Any later difference between V2-B and V2-C therefore cannot be attributed to a changed initialization function.

The branch now isolates the intended treatment:

> fully shared V2-B actor  
> → V2-B plus preference-gated private residual capacity.

## Decision

**V2-C0 PASS.**

V2-C1 modular authority / specialization screen is authorized.

V2-C1 must not judge semantic success yet. Its only questions are:
- does routing become preference-dependent?
- do expert residuals become nonzero?
- do experts receive differentiated gradients / usage?
- does masking the modular path reduce preference authority?
- does Foundation V2 remain valid?

If modular authority does not emerge, stop the V2-C branch without tuning expert count, router temperature, depth, learning rate, lambda, entropy, or foundation settings.

Primary artifacts:
- `talon_rl/v2c_actor_critic.py`
- `scripts/rl/v2c0_function_preserving_gate.py`
- `runs/v2c0_function_preserving_gate-2026-09-24/v2c0_report.json`
- `runs/v2c0_function_preserving_gate-2026-09-24/v2c0_init.pt`
- `docs/contracts/preference_architectures/v2c-contract.md`
