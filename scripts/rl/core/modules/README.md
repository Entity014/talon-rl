# Modules

<!-- nav:start -->
[RL runner](../../README.md) · [RL core](../README.md)
<!-- nav:end -->


## Purpose

Legacy/shared neural-network building blocks used by current experiments. This folder is
transitional: generic policy-facing abstractions should gradually move behind
`core/policies/` contracts without breaking old imports or checkpoint keys.

## Files

| File | Role | Status |
| --- | --- | --- |
| `actor_critic.py` | Shared squashed-Gaussian actor and vector critic | Stable legacy API |
| `env_factor_encoder.py` | Privileged environment-factor encoder | Stable utility |
| `v1a_actor_critic.py` | V1-A residual objective adapter architecture | Experimental / thesis traceable |

## Rules

Do not rename state-dict keys casually. Several function-preserving gates and checkpoints
depend on exact parameter names. New generic policy behavior should target
`core.policies.Policy`; experiment-specific architectures may remain traceable by name
until migrated deliberately.
