# Phase-1 D3 — Zero-Adaptation Sim-to-Sim Verdict

Status: **FROZEN — PARTIAL TRANSFER / SEMANTIC FAILURE**
Date: 2026-09-26

## Gate summary

    D3-A interface equivalence          PASS
    D3-B nominal dynamics transfer      PASS
    D3-C MORL semantic transfer         FAIL

Overall:

    zero-adaptation sim-to-sim          PARTIAL TRANSFER
    D4 hardware bring-up                BLOCKED

## What transferred

- exact 48-D observation/action interface;
- exact joint/actuator mapping;
- exact 20 ms action-hold contract;
- finite and stable 64-step locomotion;
- both canonical and native reset viability;
- Orientation preference semantics;
- center compromise structure;
- aggregate continuum monotonicity.

## What did not transfer

- Tracking semantics;
- Angular semantics consistently;
- continuum endpoint-envelope behavior.

Smoothness became valid in MuJoCo rather than reproducing the canonical limitation.

## Thesis interpretation

The controller transfers at the level of execution and nominal locomotion, but not at the level of complete preference semantics.

This establishes a stronger transfer hierarchy:

    interface equivalence
        !=
    dynamics viability
        !=
    semantic transfer

D2 already showed:

    pointwise action equivalence
        !=
    closed-loop deployment equivalence

D3 adds:

    deployment-equivalent execution + stable locomotion
        !=
    preservation of MORL preference semantics across physics engines

The zero-adaptation result should therefore be reported as a partial sim-to-sim transfer with semantic failure, not as a generic sim-to-sim collapse.

## Scope consequence

The thesis-critical zero-adaptation transfer gate is complete.

D4 is not authorized under the current stop rule because T/A semantic transfer is not preserved in the secondary simulator.

A later adaptation phase may study whether dynamics randomization, actuator-model calibration, or other transfer methods can recover semantics, but it must:
- preserve this D3 baseline;
- be named as a new phase;
- avoid rewriting the zero-adaptation result;
- predeclare its intervention and acceptance criteria.
