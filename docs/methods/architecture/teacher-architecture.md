# Phase 1 Teacher Architecture

The active TALON training direction is a privileged teacher that combines robot state, environment context, and a variable-cardinality objective set to realize a preference-specific locomotion policy.

## Overview

The teacher receives three sources of information:

1. robot state and previous action,
2. privileged plant/environment factors,
3. a set of objective-weight pairs.

The objective set is permutation-invariant and drives a low-dimensional learned policy family rather than only being appended to the policy observation.

## State trunk

Input:

- current state (s_t),
- previous action (a_{t-1}).

```text
[x_t, a_{t-1}] 48-D
        |
        v
Linear 48 -> 256
        |
LayerNorm / activation
        |
        v
Linear 256 -> 128
        |
activation
        |
        v
Linear 128 -> 16
        |
activation
        |
        v
h_t  (16-D)
```

## Privileged environment-factor encoder

The teacher sees the true environment factors during training:

```text
e_t =
[mass,
 CoM,
 friction,
 terrain height,
 motor strength,
 leg length,
 joint limits]
```

```text
privileged factors e_t
        |
        v
    normalize
        |
        v
Linear N_e -> 256
        |
       ELU
        |
        v
Linear 256 -> 128
        |
       ELU
        |
        v
Linear 128 -> 8
        |
        v
z_t  (8-D)
```

## Objective-set preference representation

Preference is represented as a set:

```text
{
  (o_1, w_1),
  (o_2, w_2),
  ...
  (o_m, w_m)
}
```

Example:

```text
(T, 0.5)
(A, 0.2)
(O, 0.2)
(S, 0.1)
```

Each objective receives a learned embedding (e_i), which is combined with its weight:

```text
o_i -> e_i
       |
       v
token_i = [e_i, w_i]
```

The tokens are encoded by a permutation-invariant set encoder:

```text
{token_1, ..., token_m}
        |
        v
Permutation-Invariant Set Encoder
        |
        v
preference-set latent z_w
```

Candidate implementations include DeepSets and set attention.

## Learned policy family

The preference-set latent selects a point in a learned policy family:

```text
z_w
 |
 v
Family Hypernetwork H
 |
 v
coefficients c_1 ... c_K
 |
 v
Learned Policy Bases B_1 ... B_K
 |
 v
theta(z_w) = theta_0 + sum_k c_k(z_w) B_k
```

This makes preference control act on the realized policy parameterization, not only on a concatenated observation feature.

## Conditional control

```text
h_t 16-D ----\
              +--> Conditional Policy
z_t  8-D ----/       params = theta(z_w)
                         |
                         v
                    pre-tanh u_t
                         |
                        tanh
                         |
                         v
                    action a_t 12-D
```

## Environment transition

```text
normalized action a_t
        |
        v
q_target = q_default + 0.25 a_t
        |
        v
actuator / robot dynamics
        |
        v
next state s_{t+1}
```

## Training critic

The critic receives:

- current state (x_t),
- previous action (a_{t-1}),
- privileged environment factors (e_t),
- preference-set latent (z_w).

```text
x_t
a_{t-1}
e_t
z_w
 |
 v
Shared Critic Body
 |
 v
c_t
```

Each active objective queries the same critic representation through a shared objective-query head:

```text
c_t
 |
 +----> (e_T, w_T) ----> shared objective-query head ----> V_T
 |
 +----> (e_A, w_A) ----> shared objective-query head ----> V_A
 |
 +----> (e_O, w_O) ----> shared objective-query head ----> V_O
 |
 +----> ...
```

The critic therefore follows the same variable-cardinality objective-set semantics as the actor rather than assuming one fixed output-head layout.

## Full teacher pipeline

```text
robot state s_t
previous action a_{t-1}
        |
        v
State Trunk
        |
       h_t -------------------+
                              |
privileged env factors e_t    |
        |                     |
        v                     |
Env Factor Encoder            |
        |                     |
       z_t -------------------+----> Conditional Policy
                              |        params = theta(z_w)
objective set {(o_i,w_i)}     |               |
        |                     |               v
        v                     |           pre-tanh u_t
Objective Embeddings          |               |
        |                     |              tanh
        v                     |               |
Permutation-Invariant         |               v
Set Encoder                   |           action a_t
        |                     |               |
       z_w                    |               v
        |                     |        robot/environment
        v                     |               |
Family Hypernetwork           |               v
        |                     +----------> s_{t+1}
        v
policy-family coefficients
        |
        v
theta(z_w)
```

## Intended role

This is the privileged teacher stage. Its purpose is to establish a controllable, environment-aware policy family before a later student/adaptation stage removes direct access to privileged environment factors.

Historical experiment IDs remain in experiment stages, run directories, contracts, and verdicts; this document describes the active architecture rather than replacing that provenance.
