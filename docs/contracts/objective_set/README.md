# Objective Set

<!-- nav:start -->
[Documentation](../../README.md) · [Contracts](../README.md)
<!-- nav:end -->


## Purpose

Contracts for objective-set conditioning and generalized multi-objective control.

## Scope

Covers critic substrate isolation, actor re-enable gates, credit-vs-Monte-Carlo checks, variable-cardinality objective sets, and trajectory-level semantic credit.

## Research progression

1. validate the critic substrate
2. isolate critic and actor responsibilities
3. re-enable actor updates under fixed semantics
4. test trajectory semantic credit
5. extend toward variable objective sets

## Documents

- objective-set-g1-c0-critic-isolation-contract.md
- objective-set-g1-c0-critic-substrate-contract.md
- objective-set-g1-c1-actor-reenable-contract.md
- objective-set-g1-credit-vs-mc-audit-contract.md
- objective-set-g1-trajectory-semantic-credit-contract.md
- objective-set-g1-variable-cardinality-contract.md
- objective-set-generalized-morl-phase2-contract.md

## Maintenance

Keep provenance-bearing experiment IDs in document filenames. Move documents by role/domain and update repository references when paths change.
