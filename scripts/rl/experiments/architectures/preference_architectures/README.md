# Preference Architectures

<!-- nav:start -->
[Architecture](../../../../../docs/methods/architecture/teacher-architecture.md) · [Train and run](../../../README.md) · [Experiments](../../README.md) · [Research](../../../../../docs/README.md) · [RL core](../../../core/README.md) · [Package](../../../../../talon_rl/README.md)

[TALON RL](../../../../../README.md) · [RL runner](../../../README.md) · [Experiments](../../README.md) · [Architectures](../README.md) · [Preference Architectures](README.md)
<!-- nav:end -->


Preference-conditioned policy architecture families grouped by mechanism.

- [`preference_embedding/`](preference_embedding/README.md): minimal preference embedding and authority-sensitivity gates.
- [`film_conditioning/`](film_conditioning/README.md): FiLM conditioning, update geometry, temporal credit, and retention.
- [`residual_experts/`](residual_experts/README.md): preference-gated residual expert architecture.
- [`hypernetwork/`](hypernetwork/README.md): preference-conditioned hypernetwork architecture.
- [`private_residual/`](private_residual/README.md): private residual/subspace architecture.
- [`policy_family/`](policy_family/README.md): generated policy-family parameterization.
- [`semantic_diagnostics/`](semantic_diagnostics/README.md): downstream critic, causal-semantic, gradient, and stochastic-objective diagnostics.

Legacy IDs V2A/V2B/V2C/V2H/V2K/V2PF remain in stage names and run artifacts only.
