# Models

<!-- nav:start -->
[Architecture](../../docs/methods/architecture/teacher-architecture.md) · [Train and run](../../scripts/rl/README.md) · [Experiments](../../scripts/rl/experiments/README.md) · [Research](../../docs/README.md) · [RL core](../../scripts/rl/core/README.md) · [Package](../README.md)

[TALON RL](../../README.md) · [Talon Rl](../README.md) · [Models](README.md)
<!-- nav:end -->

Reusable neural-network architectures for TALON.

This page is the local map for model code. For the active end-to-end design, read the [Teacher architecture](../../docs/methods/architecture/teacher-architecture.md).

## Current teacher direction

The active teacher decomposes into:

- **state trunk** — compact representation of robot state + previous action;
- **environment-factor encoder** — privileged plant/environment latent;
- **objective-set encoder** — permutation-invariant representation of objective/weight pairs;
- **family hypernetwork** — maps preference-set latent to policy-family coefficients;
- **conditional actor** — realizes the preference-specific policy;
- **objective-query critic** — shared critic representation queried by active objectives.

Not every item above needs its own file; the code should remain grouped by reusable mechanism.

## Existing model families

| folder | responsibility |
|---|---|
| [`foundations/`](foundations/README.md) | shared actor/critic foundations and preference embeddings |
| [`conditioning/`](conditioning/README.md) | FiLM, residual, hypernetwork, private-residual, and policy-family conditioning |
| [`behavior/`](behavior/README.md) | explicit behavior-latent and projected behavior models |
| [`authority/`](authority/README.md) | authority-isolated and objective-set-conditioned architectures |
| [`auxiliary/`](auxiliary/README.md) | training-only auxiliary model heads |

Historical identifiers such as T4, V1C, V2A, V2B, and V2H remain in class names and experiment stages for thesis traceability; they are not filesystem boundaries.

## Where to go next

- Changing the full teacher design → [Teacher architecture](../../docs/methods/architecture/teacher-architecture.md)
- Running architecture experiments → [Architecture experiments](../../scripts/rl/experiments/architectures/README.md)
- Changing objective semantics → [Rewards](../rewards/README.md)
- Changing reusable RL machinery → [RL core](../../scripts/rl/core/README.md)
