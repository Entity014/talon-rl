# Phase-1 D3-B — Zero-Adaptation Nominal Dynamics Transfer Verdict

Status: **FROZEN — PASS**
Date: 2026-09-26

## Question

Does the exact frozen D2 controller retain basic locomotion viability in MuJoCo under zero adaptation?

## Protocol

Policy:

    artifacts/phase1_canonical_actor_state.pt
    Phase1EagerStateRuntime
    CUDA
    50 Hz

MuJoCo:

    3.3.7
    dt = 0.002 s
    10 physics steps per policy action

Reset arms:

    canonical D2 joint pose
    native MuJoCo home pose

Sweep:

    5 commands
    × 5 preferences
    × 2 reset arms
    = 50 rollouts

Each rollout:

    64 policy steps
    zero adaptation

## Results

Canonical-state arm:

    rollouts                         25
    64-step survival                 25 / 25
    survival fraction                1.000
    immediate-10-step survival       1.000
    finite-state fraction            1.000
    ctrl-contract fraction           1.000
    action saturation fraction       0.9048
    median-of-median trunk height    0.2176 m
    max observed tilt                23.38 deg

Native-home arm:

    rollouts                         25
    64-step survival                 25 / 25
    survival fraction                1.000
    immediate-10-step survival       1.000
    finite-state fraction            1.000
    ctrl-contract fraction           1.000
    action saturation fraction       0.9095
    median-of-median trunk height    0.2129 m
    max observed tilt                10.42 deg

## Interpretation

The frozen controller remains dynamically viable in MuJoCo under both reset definitions.

Therefore the initial-condition mismatch between MuJoCo home and the canonical D2 pose is not sufficient to cause immediate transfer collapse.

However the transfer is stressed:

- trunk height is substantially low, around 0.21–0.22 m median;
- the actor spends roughly 90% of scalar action samples near the |a| >= 0.98 saturation criterion.

These are dynamics-transfer mismatches to report, not tuning targets within D3.

## Decision

    D3-B nominal dynamics transfer      PASS
    D3-C MORL semantic transfer         AUTHORIZED

No actuator, contact, model, policy, reward, or action-scale adaptation was used.

## Source

    docs/contracts/transfer/phase1-d3b-nominal-dynamics-contract.md
    scripts/rl/phase1_d3b_nominal_dynamics.py
    runs/phase1_d3b_nominal_dynamics/nominal_dynamics_report.json
