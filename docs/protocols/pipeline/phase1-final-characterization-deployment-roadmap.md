# Phase-1 Canonical Controller — Final Characterization and Deployment Roadmap

Status: **ACTIVE THESIS-CRITICAL PATH**
Date: 2026-09-25

## Canonical controller

The Phase-1 deployment candidate is the authority-isolated u30 controller:

    runs/authority_isolated_simplex_edge_retain-2026-09-25/model_30.pt

This checkpoint is the canonical reference for all final characterization and deployment work unless a later document explicitly supersedes it.

## Frozen scientific status

Validated:

    preference authority              PASS
    authority durability              PASS
    critic compatibility              PASS
    robustness substrate              PASS
    T semantic validity               PASS
    A semantic validity               PASS
    O semantic validity               PASS

Documented limitation:

    S semantic validity               INCONSISTENT

Not claimed:

    4/4 semantic validity
    semantic forgetting solved
    H2b semantic durability
    objective generalization

Recommended final-method wording:

> The final Phase-1 method substantially improves preference authority, authority durability, robustness, and simultaneous semantic validity across three of four objectives, while Smoothness remains an unresolved heterogeneous closed-loop semantic limitation.
## Stage D0 — canonical freeze and provenance

Before new training or deployment:
- freeze model hash;
- freeze observation ordering;
- freeze action ordering;
- freeze preference ordering T/A/O/S;
- freeze reward/semantic definitions;
- freeze normalization constants;
- freeze deterministic inference path;
- freeze actuator/action scaling;
- record software/IsaacLab/PyTorch environment.

No Phase-1 architecture or reward modification is allowed after D0.

## Stage D1 — multi-seed final characterization

Purpose:

> Determine whether the final Phase-1 formulation reproducibly preserves its validated qualitative result across independent training seeds.

This is characterization, not method selection.

Primary questions:
1. Is preference authority reproducible?
2. Is authority durability reproducible?
3. Are T/A/O semantics reproducible?
4. Does robustness remain bounded?
5. Does S remain a scoped semantic limitation rather than revealing a new engineering failure?

Do not require S to PASS as an acceptance condition.

Use preregistered independent seeds and fixed training budget.
No poor seed may be replaced.
No checkpoint may be selected using endpoint semantic scores.
## Stage D2 — deployment-oriented simulation validation

Use the canonical controller contract and test:

### Runtime
- deterministic action repeatability;
- inference latency;
- control-loop jitter;
- stale-observation handling;
- finite-value assertions.

### Observation contract
- joint order;
- quaternion convention;
- body-frame velocities;
- projected gravity;
- command scaling;
- action-history term;
- normalization.

### Action contract
- joint order;
- nominal pose offsets;
- action scale;
- saturation;
- rate limits;
- actuator-safe range.

### Safety
- fall detection;
- base-contact termination;
- command timeout;
- estop;
- startup/zero-command state;
- recovery behavior;
- preference bounds.

### Semantic sanity
Replay T/A/O/S preference endpoints and center under deployment wrapper.
T/A/O must retain the frozen simulation semantics within preregistered tolerance.
S remains reported as a limitation.

Failure here is a deployment-interface failure, not a MORL-method failure.
## Stage D3 — sim-to-sim validation

**FROZEN RESULT (2026-09-26): PARTIAL TRANSFER / SEMANTIC FAILURE**

    D3-A interface equivalence      PASS
    D3-B nominal dynamics           PASS
    D3-C MORL semantic transfer     FAIL

The exact D2 deployment boundary executes correctly in MuJoCo and survives nominal zero-adaptation rollouts, but T/A preference semantics and continuum endpoint-envelope behavior do not transfer reliably. Orientation transfers; Smoothness is valid in the MuJoCo regime.

Canonical source:

    docs/verdicts/transfer/phase1-d3-zero-adaptation-sim2sim-verdict.md

Under the frozen D3 stop rule, D4 is blocked. Any transfer adaptation must be opened as a new explicitly named phase against this preserved zero-adaptation baseline.

## Stage D4 — guarded hardware bring-up

**BLOCKED by D3-C semantic-transfer failure.**

Order:

    1. offline inference replay
    2. robot powered, motors disabled
    3. motor enable with zero/nominal command
    4. supported/secured low-gain test
    5. short flat-ground locomotion
    6. preference endpoint tests
    7. center/interior preference tests
    8. longer autonomous runs

Every stage requires explicit safety exit criteria before advancing.

No online RL training on hardware is part of the current thesis path.
## Stage D5 — real-robot evaluation

Primary scientific evaluation should remain aligned with Phase 1:

- T preference response;
- A preference response;
- O preference response;
- S reported honestly;
- center compromise;
- command tracking;
- survival/fall rate;
- energy/current/torque if available;
- preference-action separation;
- repeatability across runs.

The hardware experiment tests transfer of the frozen controller behavior.

It does not reopen Phase-1 method selection.

## Thesis scope after V3 closure

Main thesis contribution:

    reliable fixed-vocabulary preference-conditioned MORL
    + causal authority/semantic validation
    + deployment

Extension result:

    objective-set representation generalizes structurally
    but variable-set semantic learning is not established

Future work:

    semantically stable objective-set learning
    compositional generalization (G2)
    unseen-objective representation (G3)

## Immediate next artifact

The next experiment contract should be:

    Phase-1 Multi-Seed Final Characterization Contract

Only after that contract is frozen should new Phase-1 training seeds be launched.
