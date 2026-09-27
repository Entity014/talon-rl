# Phase 1 Teacher Architecture (V4)

<!-- nav:start -->
[Architecture](teacher-architecture.md) · [Train and run](../../../scripts/rl/README.md) · [Experiments](../../../scripts/rl/experiments/README.md) · [Research](../../README.md) · [RL core](../../../scripts/rl/core/README.md) · [Package](../../../talon_rl/README.md)

[TALON RL](../../../README.md) · [Documentation](../../README.md) · [Methods](../README.md) · [Architecture](README.md)
<!-- nav:end -->

The active TALON training direction is a privileged teacher. It combines
robot state, plant context, and a variable-cardinality objective set to
realize a preference-specific locomotion policy.

This page describes what is implemented, in
`talon_rl/models/authority/teacher_v4.py` (`TeacherV4`), as trained and
evaluated in V4-C. Results:
[V4-C closure](../../verdicts/teacher_v4/teacher-v4-c-closure.md).

## Overview

The teacher receives three inputs:

1. the canonical 48-D observation (state, command, previous action);
2. the 12-D privileged plant context e_t, normalized outside the model;
3. a set of objective–weight pairs {(o_i, w_i)}.

The objective set is permutation-invariant. It reaches the actor only
through a low-dimensional family of residual weights on the last two policy
layers; it never enters the state trunk or the plant encoder.

## 1. State trunk

```text
canonical observation x_t, 48-D
(base lin/ang vel, projected gravity, command, joint pos/vel, previous action)
        │
        ▼
Linear 48 → 256, LayerNorm, ELU
Linear 256 → 128, ELU
Linear 128 → 16, ELU
        │
        ▼
h_t  (16-D)
```

## 2. Privileged plant context

e_t is 12-D, in this frozen order: static friction (1), motor Kp and Kd (2),
leg length (1), joint range (1), terrain height (1), dynamic friction (1),
passive joint damping (1), payload mass and CoM (4). Contract:
[input-contract verdict](../../verdicts/teacher_v4/teacher-v4-a-b-input-contract-verdict.md).

```text
raw e_t, 12-D
        │
        ▼
RunningNormalizer (centered; outside TeacherV4;
statistics saved with the checkpoint)
        │
        ▼
Env Encoder
Linear 12 → 256, ELU
Linear 256 → 128, ELU
Linear 128 → 8
        │
        ▼
z_t  (8-D)
```

## 3. Objective-set preference representation

The preference is a set of objective IDs with weights on the simplex:

```text
{ (o_1, w_1), …, (o_m, w_m) },   Σ w_i = 1,   m = 1…4
e.g. (T, 0.5), (A, 0.2), (O, 0.2), (S, 0.1)
```

Weighted DeepSets encoder:

```text
o_i ──► E(o_i)   objective embedding, 16-D
            │
            ▼
        φ: Linear 16 → 64, ELU, Linear 64 → 64
            │
            × w_i
            ▼
        Σ_i w_i φ(E(o_i))
            │
            ▼
        ρ: ELU, Linear 64 → 64, ELU, Linear 64 → 16
            │
            ▼
        z_w  (16-D)
```

```text
z_w = ρ( Σ_i w_i φ(E(o_i)) )
```

The weight multiplies the embedded objective, so the sum is permutation
invariant by construction, and w_i = 0 contributes exactly nothing. A
zero-weight entry is therefore exact padding.

## 4. Learned policy family

```text
z_w
 │
 ▼
Family hypernetwork H: Linear 16 → 32, ELU, Linear 32 → 8
 │
 ▼
coefficients c_1 … c_8
 │
 ▼
θ(z_w) = θ_0 + Σ_k c_k(z_w) B_k
```

θ(z_w) covers only the last two policy layers (256 → 128 → 12). It is not the
whole actor.

## 5. Conditional actor

```text
h_t (16-D) ──┐
             ├── concat, 24-D
z_t (8-D) ───┘
        │
        ▼
shared backbone: Linear 24 → 256, ELU          (preference-independent)
        │
        ▼
family residual layers, weights θ_0 + Σ_k c_k(z_w) B_k:
    256 → 128, ELU
    128 → 12
        │
        ▼
pre-tanh mean u_t;  Gaussian with learned log-std
        │
       tanh
        │
        ▼
normalized action a_t, 12-D, in [−1, 1]
```

The preference changes only the realized end of the policy. The state trunk,
the env encoder and the 24 → 256 backbone see no preference.

## 6. Environment transition (V4-C substrate = stock A1 flat, M0)

```text
a_t
 │
 ▼
q_target = q_default + 0.25 a_t
 │
 ▼
DC-motor PD actuator, Kp = 25, Kd = 0.5
 │
 ▼
physics: dt = 0.005 s, decimation 4 (policy step 0.02 s)
 │
 ▼
x_{t+1}, e_{t+1}
```

## 7. Critic

The critic has its own objective embedding and set encoder, so a value
loss can never move the actor's preference representation. Gradient
isolation is tested in both directions.

```text
x_t (48-D) + normalized e_t (12-D) + z_w^V (16-D, critic's own set encoder)
        │
        ▼
Shared critic body: Linear 76 → 256, ELU, 256 → 256, ELU, 256 → 128, ELU
        │
        ▼
c_t  (128-D)

objective ID o_i ──► critic objective embedding ──► q_i (16-D)

[c_t, q_i] ──► shared query MLP: Linear 144 → 64, ELU, 64 → 1 ──► V_i
```

```text
c_t = f_V(x_t, e_t, z_w^V),   q_i = E_critic(o_i),   V_i = Q_V([c_t, q_i])
```

w_i does not enter the query head. The whole preference composition is
already in z_w^V. Each active objective is queried through the same head,
so the critic follows the same variable-cardinality semantics as the actor.

## 8. Training (V4-C)

Objective-set PPO inside the M0 (stock `rsl_rl`) shell:
`scripts/rl/core/algorithms/objective_set_ppo.py`. The actor loss is
m · Σ_i w_i · clipped-surrogate(A_i) over the active set. The critic loss is
a per-objective clipped value loss. Actor and critic use separate Adam
optimizers with the shared adaptive-KL LR. 4096 envs × 24 steps, 5 epochs ×
4 minibatches, 29.49M samples. Contract:
[rollout/batch audit](../../verdicts/teacher_v4/teacher-v4-c0-rollout-batch-audit.md).

## Full teacher pipeline

```text
x_t (48-D) ──► State Trunk ──► h_t (16) ──┐
                                          ├─► 24 → 256 (shared) ─► family residual 256 → 128 → 12 ─► tanh ─► a_t
raw e_t ─► normalizer ─► Env Encoder ─► z_t (8) ─┘                        ▲
                                                                          │ θ_0 + Σ c_k B_k
{(o_i, w_i)} ─► E_actor, φ, ×w_i, Σ, ρ ─► z_w (16) ─► hypernetwork ─► c_1…c_8

critic: [x_t, e_t, z_w^V] ─► c_t;  [c_t, q_i] ─► V_i   (separate E_critic and set encoder)
```

## What is not in the architecture yet

**Semantic preservation.** V4 guarantees preference authority: different
weights give different policies. Nothing in the model or loss guarantees
that raising w_i moves behavior in objective i's intended direction. V4-C
showed the gap: authority is strong in every run and T/O/S semantics are
reproducible, but A fails systematically (2/10 seeds) and the A and S critics
are invalid. V4-C2 (branch `v4-c2-semantic-preservation`) designs that
mechanism:
[semantic-relation contract](../../contracts/teacher_v4/teacher-v4-c2-semantic-relation-contract.md).

## Intended role

This is the privileged teacher stage. Its purpose is to establish a
controllable, plant-aware policy family before a later student/adaptation
stage removes direct access to e_t.

Historical experiment IDs remain in experiment stages, run directories,
contracts, and verdicts. This page describes the active architecture; it
does not replace that provenance.
