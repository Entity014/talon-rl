# Algorithms

<!-- nav:start -->
[Architecture](../../../../docs/methods/architecture/teacher-architecture.md) · [Train and run](../../README.md) · [Experiments](../../experiments/README.md) · [Research](../../../../docs/README.md) · [RL core](../README.md) · [Package](../../../../talon_rl/README.md)

[TALON RL](../../../../README.md) · [RL runner](../../README.md) · [RL core](../README.md) · [Algorithms](README.md)
<!-- nav:end -->


## Purpose

Own training orchestration and algorithm-family-specific update semantics. Files are grouped
by algorithm responsibility rather than thesis experiment chronology.

## Files

| File | Role | Status |
| --- | --- | --- |
| `__init__.py` | Stable `Trainer` Protocol and public exports | Stable |
| `scalar_ppo.py` | Scalar PPO family: B0 baseline + L0 reference recipe | Research baseline |
| `vector_ppo.py` | Vector/preference PPO primitives + function-preserving variants | Experimental |
| `amor.py` | AMOR early-scalarization family + scale-aligned variant | Experimental |
| `moppo.py` | Full preference-conditioned MOPPO trainer/config | Transitional; monolithic |

## Dependency Rules

Algorithms may depend on policies, objectives, rollout, preferences, normalization, and
checkpoint services. Lower-level components must not import concrete trainers.

## Naming Rule

Keep experiment identity in class/function names, configs, tests, and experiment scripts when
it matters for thesis traceability. Reusable module filenames should describe algorithm
semantics rather than chronology such as M0.2A or M0.3B.

## Adding an Algorithm

Add a new module only when the algorithm has genuinely different update semantics. Small
variants of the same family should normally live in the family module. Prefer composition
over subclass chains and keep experiment-only parameter choices outside core.
