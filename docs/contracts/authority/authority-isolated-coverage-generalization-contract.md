# Authority-Isolated Coverage / Generalization Branch Contract

Status: **FROZEN BEFORE EXECUTION**
Date: 2026-09-25

## Hypothesis

The remaining robustness failures reflect insufficient coverage of reset-state and dynamics regimes during training rather than a missing residual-specific controller mechanism.

## Starting point

Both arms resume from the exact same repaired-u10 state:
- identical actor
- identical wide critic
- identical optimizer state
- identical adaptive critic-support pools
- identical anchor specs
- identical RNG state snapshot before continuation

Frozen method contract:
- authority-isolated actor
- continuous one-model preference w
- wide critic
- GAE lambda = .95
- projected PPO/tail-conflict repair
- active tail descent kappa = .05
- MORL reward/objective semantics
- PPO ratio contract
- evaluation semantics

## Sole treatment

CONTROL:
- continue repaired-u10 under the original UnitreeA1FlatEnvCfg training distribution

COVERAGE:
- same continuation, but broaden training distribution only

Coverage changes:
1. root reset pose:
   - retain x/y/yaw ranges
   - add z in [-0.03, +0.03] m
   - add roll/pitch in [-0.12, +0.12] rad

2. root reset velocity:
   - widen each linear/angular component from +/-0.5 to +/-0.75

3. joint velocity:
   - retain original joint-position reset
   - after that reset, add joint velocity offsets sampled uniformly in [-1.5, +1.5] rad/s

4. contact dynamics:
   - resample rigid-body material every reset
   - static friction in [0.6, 1.4]
   - dynamic friction in [0.5, 1.2]
   - restitution in [0.0, 0.1]

No external push term is added. The existing 10-15 s interval push is replaced in the coverage arm by the joint-velocity reset event because the short 32-64 step training rollouts do not reach the interval anyway.

No reward, action parameterization, preference schedule, critic architecture, or actor objective is changed.

## Screen

Paired continuation:

    start update = 10
    end update   = 20

Same seed and same update count.

## Primary evaluation

Frozen semantic suites:
- suite2 = seed 840003
- suite3 = seed 840004

All T/A/O/S/C preferences, 8 lanes each.

Primary gate:

    min semantic-suite survival >= .95

## Held-out generalization evaluation

Nominal original evaluation environment, unseen reset seeds:

    suite4 = 850101
    suite5 = 850202
    suite6 = 850303

The held-out suites are not used during training or treatment construction.

Required:
- treatment must improve or preserve aggregate held-out survival relative to control
- improvement cannot be restricted only to broadened-training reset draws

## No-regression gates

Authority:
- pairwise action separation retention >= .90 from repaired-u10 start
- simplex-tangent Jacobian retention >= .90 from repaired-u10 start

Critic:
- wide critic fresh H32 / MC64 validity must remain bounded and not collapse

PPO:
- pre-update ratio invariant remains valid

Termination:
- no new systematic termination pattern

## Decision

PASS:
- min semantic-suite survival >= .95
- authority gates pass
- critic gate passes
- held-out reset generalization is not worse than control
- no new systematic failure topology

FAIL:
- coverage broadening does not improve held-out/frozen-suite robustness sufficiently
- or gains come with authority/critic regressions

A PASS authorizes return to AI-C2, then AI-H2.

A FAIL closes simple coverage-deficit repair and motivates deployment-level safety/recovery mechanisms rather than further residual-specific mechanism mining.
