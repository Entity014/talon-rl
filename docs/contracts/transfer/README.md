# Transfer

<!-- nav:start -->
[Documentation](../../README.md) · [Contracts](../README.md)
<!-- nav:end -->


## Purpose

Contracts for deployment, simulator transfer, plant alignment, controller robustness, and ensemble robustness.

## Scope

Covers deployment equivalence, zero-adaptation sim-to-sim transfer, plant/actuator matching, controller transfer sensitivity, and ensemble-training validity.

## Research progression

1. freeze deployment semantics
2. establish interface and runtime equivalence
3. characterize plant mismatch
4. repair or bound simulator mismatch
5. test controller robustness
6. evaluate ensemble training without losing source semantics

## Documents

- phase1-d1-multiseed-final-characterization-contract.md
- phase1-d2-deployment-equivalence-contract.md
- phase1-d3-zero-adaptation-sim2sim-contract.md
- phase1-d3b-nominal-dynamics-contract.md
- phase1-d3c-semantic-transfer-contract.md
- phase1-semantic-target-design-contract.md
- phase3-t1-semantic-transfer-mechanism-audit-contract.md
- phase3-t2-actuator-calibration-contract.md
- phase3-t3-explicit-dcmotor-emulation-contract.md
- phase3-t4-passive-joint-correction-contract.md
- phase3-t4-readonly-plant-equivalence-audit-contract.md
- phase4-controller-transfer-robustness-audit-contract.md
- phase5-e1-frozen-policy-validity-envelope-contract.md
- phase5-e2-training-screen-evaluation-contract.md
- phase5-plant-ensemble-semantic-robustness-contract.md

## Maintenance

Keep provenance-bearing experiment IDs in document filenames. Move documents by role/domain and update repository references when paths change.
