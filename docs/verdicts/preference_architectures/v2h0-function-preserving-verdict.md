# V2-H0 Function-Preserving Gate Verdict

Status: **FROZEN — V2-H0 PASS; V2-H1 AUTHORIZED**

Date: 2026-09-24

## Scope

V2-H tests a preference-conditioned policy-parameter manifold rather than feature modulation, expert routing, or retention.

Treatment only:

    c(w) = HyperNet(w)
    B_k = U_k V_k
    W(w) = W0 + sum_k c_k(w) B_k

The generated low-rank delta is applied only to the actor action-head weight.

Configuration:
- K = 4 latent bases
- rank = 4 per basis
- coefficient network = 4 -> 16 -> 4
- bases are unlabeled latent policy directions
- coefficient output layer is zero-initialized
- U and V bases are nonzero at initialization

Therefore:

> V2-H(initial) ≡ V2-B(initial)

exactly for every preference.

## Gate result

All H0 invariants pass.

### Exact policy identity

- deterministic action max absolute difference vs V2-B: **0.0**
- pre-tanh mean max absolute difference: **0.0**
- same-latent transformed log-probability difference: **0.0**
- PPO ratio max |ratio - 1|: **0.0**

### Critic / environment identity

- critic value difference: **0.0**
- environment raw-action difference: **0.0**
- 64-step paired no-update smoke max action difference: **0.0**
- termination events: **0**

### Hypernetwork identity initialization

- coefficient max absolute value: **0.0**
- generated weight-delta max absolute value: **0.0**

The latent bases are not dead:
- hyper-U norm = **0.2000**
- hyper-V norm = **0.4000**

This initialization therefore differs from a zero-capacity branch: the policy is exactly preserved, but gradients can immediately move the coefficient mapping because the latent weight bases already span nonzero directions.

### Treatment isolation

All existing V2-B tensors are copied exactly:
- maximum pre-existing parameter difference = **0.0**

The only new trainable parameters are:
- hyper_u
- hyper_v
- preference_hyper first-layer weight/bias
- preference_hyper output-layer weight/bias

No critic, V2-B actor, embedding, FiLM, PPO, reward, support, or evaluation parameter is altered.

### Checkpoint integrity

- action roundtrip difference: **0.0**
- value roundtrip difference: **0.0**
- full state hash equality: **PASS**

### Objective contract

The frozen normalized four-objective probe and scalarization contract pass exactly.

## Interpretation

V2-H0 establishes an exact causal starting point for the policy-manifold hypothesis.

The branch differs from previous architectures conceptually:

- V2-A changes preference representation;
- V2-B changes hidden features through FiLM;
- V2-C attempts preference-conditioned expert routing;
- V2-H allows preference to generate a distinct low-rank action-head parameter realization.

At H0, none of this additional parameter authority is active yet.

Any later difference therefore arises from learned preference-conditioned weight generation rather than from a changed initial policy.

## Decision

**V2-H0 PASS.**

V2-H1 parameter-manifold authority screen is authorized.

H1 must not judge semantic endpoint success.

Its only questions are:
- do T/A/O/S-heavy preferences generate distinct coefficient vectors?
- do they generate distinct effective weight deltas?
- does the generated manifold acquire nontrivial effective rank?
- does masking the generated delta reduce preference-conditioned action authority?
- do coefficient/basis gradients remain active?
- does Foundation V2 remain valid?

If H1 fails to establish distinct policy-parameter realizations with causal action authority, stop the branch without tuning basis count, rank, coefficient-network depth, learning rate, lambda, entropy, or foundation settings.

Primary artifacts:
- `talon_rl/v2h_actor_critic.py`
- `scripts/rl/v2h0_function_preserving_gate.py`
- `runs/v2h0_function_preserving_gate-2026-09-24/v2h0_report.json`
- `runs/v2h0_function_preserving_gate-2026-09-24/v2h0_init.pt`
- `docs/contracts/preference_architectures/v2h-contract.md`
