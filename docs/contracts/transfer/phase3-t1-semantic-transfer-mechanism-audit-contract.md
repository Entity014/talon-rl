# Phase 3 T1 — Read-Only Semantic Transfer Mechanism Audit Contract

Status: **PREDECLARED — READ ONLY**
Date: 2026-09-26

## Frozen baseline

D3 zero-adaptation remains immutable:

    D3-A interface equivalence      PASS
    D3-B nominal dynamics           PASS
    D3-C semantic transfer          FAIL

T1 does not modify policy, simulator, reward, action scale, gains, contacts or friction.

## Primary question

Why does T-heavy reverse relative to center in MuJoCo even though the exact deployment controller remains alive and preference-responsive?

## Matched protocol

Engines:

    IsaacLab / PhysX
    MuJoCo 3.3.7

Controller:

    artifacts/phase1_canonical_actor_state.pt
    Phase1EagerStateRuntime
    CUDA
    50 Hz

Initial state in both engines:

    root pose      [0,0,0.43, 1,0,0,0]
    root velocity  zero
    joint position canonical D2 default
    joint velocity zero
    previous action zero

Isaac audit disables observation corruption, heading-command post-processing and standing sampling. This is an isolation audit, not a replacement for the frozen D3 baseline.

Commands:

    forward      [0.5, 0.0,  0.0]
    turn_left    [0.3, 0.0,  0.3]
    turn_right   [0.3, 0.0, -0.3]
    lateral      [0.0, 0.25, 0.0]

Preferences:

    T-heavy [0.7,0.1,0.1,0.1]
    center  [0.25,0.25,0.25,0.25]

64 policy steps per rollout.

## Logged trajectory channels

Per step:

    base vx/vy/vz
    base wx/wy/wz
    projected gravity
    tracking error
    T weighted reward / normalized T objective
    cumulative T return
    joint q / qdot
    action
    action rate
    height
    tilt
    contact count
    action saturation fraction

## Derived matched quantities

Per engine and command:

    Delta a(t)          = a_T(t) - a_C(t)
    ||Delta a(t)||
    Delta tracking(t)   = error_T - error_C
    Delta vx(t)
    Delta wz(t)
    cumulative Delta J_T(0:t)
    state divergence
    joint-state divergence
    contact timing difference

Cross-engine:

    center baseline shift
    T-heavy baseline shift
    action-separation ratio MuJoCo/Isaac
    saturation shift
    height shift

## Interpretation

A) effective policy response changes:
   T-vs-center action separation or direction changes strongly after state visitation diverges.

B) same preference action separation, different dynamics effect:
   action authority remains comparable but T-vs-center tracking effect flips across engines.

C) center baseline shift:
   center enters a substantially different operating regime that changes the relative T-heavy comparison.

D) actuator/contact/saturation regime:
   semantic reversal co-occurs with strong saturation, low height, or contact-timing changes.

No single label is forced. Multiple mechanisms may contribute.

## Stop rule

T1 authorizes no adaptation by itself.

Only if a recurrent, interpretable transfer mechanism is identified may T2 predeclare one minimal adaptation candidate.
