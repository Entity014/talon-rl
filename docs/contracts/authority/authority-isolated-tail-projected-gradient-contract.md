# Authority-Isolated Tail-Projected Gradient Contract

Status: **FROZEN BEFORE EXECUTION**
Date: 2026-09-25

## Motivation

The rho=0.10 gradient-budgeted screen preserved authority at update 3 but did not control tail growth. PPO and tail gradients were strongly conflicting:
- update 1 cosine = -0.757
- update 2 cosine = -0.529
- update 3 cosine = -0.187

A fixed small tail budget cannot prevent first-order tail increase when the PPO gradient contains a large anti-tail component.

## Sole treatment

Compute g_ppo and g_tail separately.

If dot(g_ppo, g_tail) < 0, remove only the PPO component that would increase tail under gradient descent:

    g_safe = g_ppo - [dot(g_ppo,g_tail) / (||g_tail||^2 + eps)] g_tail

Otherwise:

    g_safe = g_ppo

No extra tail-descent component is added in this first screen.

This is the minimum-norm modification that enforces:

    dot(g_safe, g_tail) >= 0

so the first-order update is not allowed to increase L_tail.

Then apply the same global gradient clip=1.0 and Adam step.

## Frozen components

Everything else remains identical to the paired control:
authority-isolated actor, wide critic, GAE lambda=.95, Foundation support, preference schedule, tau_j, optimizer/lr, stochastic PPO contract.

## Initial screen

3 updates, seed 73001.

## Gates

- authority pairwise retention >= .90
- tangent Jacobian retention >= .90
- probe tail loss/fraction must not show uncontrolled growth
- report projection fraction and removed PPO-gradient norm
- no AI-C2/H2 claim
