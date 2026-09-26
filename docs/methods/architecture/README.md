# Architecture

<!-- nav:start -->
[Documentation](../../README.md) · [Methods](../README.md)
<!-- nav:end -->


## Purpose

Architecture-level method notes that define reusable model structure beyond one individual experiment.

## Scope

This folder owns the active teacher architecture and architecture amendments that inform multiple downstream experiments. Formal pass/fail criteria still belong under `docs/contracts/`, and completed conclusions belong under `docs/verdicts/`.

## Current architecture

- `teacher-architecture.md` — canonical Phase 1 privileged teacher design: state trunk, environment-factor encoder, objective-set encoder, learned policy family, conditional policy, and shared objective-query critic.
- `foundation-amendment-v2.md` — retained foundation amendment from the preceding architecture research chain.

## Research progression

1. establish a reusable policy/critic foundation;
2. preserve explicit preference authority;
3. condition on privileged plant/environment context during teacher training;
4. represent objectives as a permutation-invariant set;
5. realize preference-specific policies through a learned policy family;
6. later distill/adapt away privileged environment access.

## Related code

- `talon_rl/models/` — reusable model architectures.
- `scripts/rl/core/modules/` — generic RL modules/encoders.
- `scripts/rl/experiments/architectures/` — architecture research workflows.

## Maintenance

Keep architecture descriptions mechanism-focused. Historical experiment IDs stay in contracts, verdicts, stages, and run artifacts rather than becoming the primary architecture names here.
