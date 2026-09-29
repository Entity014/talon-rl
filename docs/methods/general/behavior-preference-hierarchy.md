# Behavior, Task Requirement and Preference Objectives (working design)

<!-- nav:start -->
[Architecture](../architecture/teacher-architecture.md) · [Train and run](../../../scripts/rl/README.md) · [Experiments](../../../scripts/rl/experiments/README.md) · [Research](../../README.md) · [RL core](../../../scripts/rl/core/README.md) · [Package](../../../talon_rl/README.md)

[TALON RL](../../../README.md) · [Documentation](../../README.md) · [Methods](../README.md) · [General](README.md)
<!-- nav:end -->

Status: **working design, 2026-09-29.** It replaces the "every term competes
on one simplex" formulation used from V4-C to F8. Nothing here is trained
yet.

## Why

From V4-C to F8, tracking (T) was one objective on the same simplex as the
stability axes. Standing is a natural optimum of every stability axis, so
most of the simplex preferred not walking (V4-C 15/16 centers stand, F8
base not viable). The [FA audit](../../verdicts/teacher_v4/teacher-v4-fa-formulation-audit.md)
showed a tension that a single simplex cannot resolve. T3-B scaling gives
the stability axes authority but makes standing win. Stock scaling makes
walking win but removes their authority. Inside the locomoting set, R, V
and O genuinely trade off.

## Principle

**The behavior decides what must be done; the preference decides how.**
Having to walk is not a preference. How to walk is.

    Behavior                    e.g. LOCOMOTION, STAND, RECOVERY
    ├── task requirement        must be met, never traded (locomotion: velocity tracking)
    ├── sub-behavior / command  what it is doing now (vx, yaw rate; terrain later)
    └── preference objectives   traded by the user, inside feasible behavior
        └── objective → sub-objective → reward term → raw measurement

Policy: π(a_t | s_t, e_t, b_t, c_t, w_t). b is the behavior, c the command,
and w the preference **of that behavior only**. Behaviors never share one
weight vector: [walk, stand, recover, R, V, O] is never a simplex. Each
behavior has its own requirement and its own preference set, which may be
empty (e.g. recovery).

Not every measurement becomes an objective. The role screen still assigns
constraints (e.g. foot impact), regularizers (e.g. joint power) and gait
priors. They are never user-facing. Users see behavior, command and
preference, never raw reward terms.

## Locomotion (current phase)

    LOCOMOTION
    ├── requirement: tracking (linear xy + yaw rate)
    ├── command: vx, vy, yaw rate
    └── preferences: w_R + w_V + w_O (+ w_E if selection supports it) = 1
        R  rotational stability   ang_vel_xy_l2
        V  vertical stability     lin_vel_z_l2 or body_height_osc_l2 (F8 unresolved)
        O  orientation            flat_orientation_l2
        E  efficiency/smoothness  candidate only

Plan: prove the hierarchy on locomotion alone, then add behaviors (stand,
recovery, …). The long-term target is one policy that samples behavior,
command and preference per episode.

## Open design questions (for the next contract)

1. **How the requirement enters training:** a fixed-weight task term
   λ_T · T + wᵀR_pref; a constraint T ≥ T_min (Lagrangian); or lexicographic
   (preferences act only once tracking is met).
2. **The task-vs-preference scale.** With a fixed task term, the FA tension
   moves into λ_T. It must be set by a pre-declared feasibility criterion,
   for example locomotion at every vertex of the preference simplex, not
   tuned on semantic results.
3. **The critic:** T as its own value stream outside the preference set,
   with the objective-set encoder over R, V, O only.
4. **Normalization inside the preference block:** R, V, O relative to each
   other only (T3-B among themselves or otherwise), now that T is not on the
   simplex.
