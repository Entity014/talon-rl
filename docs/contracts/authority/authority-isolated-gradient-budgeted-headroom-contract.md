# Authority-Isolated Gradient-Budgeted Headroom Contract

Status: **FROZEN BEFORE EXECUTION**
Date: 2026-09-25

## Motivation

The fixed-coefficient treatment lambda_tail=0.01 failed. At update 0 its effective tail-gradient norm was about 0.0083 versus an observed PPO actor-gradient norm about 14.6, only ~0.057%. Tail pressure therefore had essentially no optimization authority.

The tail gradient is not intrinsically opposed to fixed-probe preference separation at update 0:
- full cosine(tail, authority-increasing gradient) = +0.226
- actor trunk = +0.196
- family path = +0.255

## Sole new treatment

Keep the same per-coordinate robust-u50 tail budgets tau_j and same tail loss.

Compute PPO and tail gradients separately on each actor update:

    g_ppo  = grad(L_PPO)
    g_tail = grad(L_tail)

Scale tail gradient to a fixed bounded fraction rho of PPO gradient norm:

    s = rho * ||g_ppo|| / (||g_tail|| + eps)
    g = g_ppo + s g_tail

with:

    rho = 0.10

Then apply the same global actor gradient clip (1.0) and Adam optimizer step as the control.

No fixed lambda is used. No residual-lane data enter the gradient scale.

## Frozen components

- authority-isolated actor: same
- wide critic: same
- GAE lambda=0.95
- Foundation/reset-diverse support: same
- preference schedule: same
- tau_j: same robust-u50 fixed-probe q95 budgets
- optimizer/lr: same
- stochastic PPO action/log-prob path: same

## Diagnostics per update

- PPO gradient norm
- raw tail gradient norm
- applied tail scale s
- scaled tail/PPO norm ratio
- cosine(g_ppo,g_tail)
- combined preclip norm
- tail loss/fraction
- termination fraction

## Gates

Same as the coordinate-aware headroom contract:
- pairwise preference authority retention >= 0.90
- tangent Jacobian retention >= 0.90
- tail mechanism improves relative to update 0
- suite3/C/lane0 survives without losing FL-hip support margin
- suites2/3 min survival >= 0.95
- failed lanes lower than control

This is a mechanism screen, not AI-C2/H2.
