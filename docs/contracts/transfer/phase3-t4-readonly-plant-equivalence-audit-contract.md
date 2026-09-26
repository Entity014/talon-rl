# Phase 3 — Read-Only Plant / Contact Equivalence Audit Contract

Status: **PREDECLARED — READ ONLY**
Date: 2026-09-26

## Purpose

Localize the residual source-target plant mismatch remaining after T2/T3 actuator-side hypotheses.

Frozen evidence:

    D3 zero-adaptation semantic transfer      FAIL
    T1 dynamics-driven state-visitation       SUPPORTED
    T2 nominal PD calibration                 FAIL
    T3 explicit DCMotor emulation             materially helpful, insufficient

No parameter fitting, policy tuning, reward change, actuator retuning, contact tuning or threshold change is permitted in this audit.

## Audit groups

A. Inertial equivalence
- link mass
- center of mass / inertial frame
- inertia tensor
- joint armature / effective inertia
- total robot mass

B. Passive joint dynamics
- damping
- friction / frictionloss
- reflected rotor inertia / armature
- joint limits

C. Contact / solver dynamics
- ground/foot friction
- restitution / compliance
- contact stiffness / damping representation
- solver type / iterations
- timestep / substep convention

## Policy-free probes

1. free response / drop
2. single-joint torque pulse
3. matched torque impulse
4. stance load response
5. foot contact impulse
6. small-angle body perturbation

Primary comparison quantities:

    Delta qdot
    Delta joint acceleration
    Delta base angular velocity
    Delta COM linear velocity
    contact impulse
    settling / damping behavior

## Decision logic

If inertial mismatch dominates:
    authorize one T4 inertial/armature correction candidate.

If passive-joint mismatch dominates:
    authorize one T4 damping/friction correction candidate.

If contact/solver mismatch dominates:
    authorize one T4 contact/solver correction candidate.

If no single group dominates:
    freeze distributed plant mismatch.
    No single T4 candidate is authorized.

## Governance

The T3 canonical height-floor failures are near-boundary failures and remain reported exactly as frozen.

This audit may explain them but may not change the 0.18 m threshold retrospectively.
