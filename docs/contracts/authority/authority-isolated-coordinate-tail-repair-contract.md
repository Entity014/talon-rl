# Authority-Isolated Coordinate-Aware Tail Repair Contract

Status: **FROZEN BEFORE TRAINING**
Date: 2026-09-25

## Frozen components

Both paired arms keep identical:
- authority-isolated actor architecture
- wide critic architecture / initialization
- GAE lambda = 0.95
- preference schedule
- Foundation anchor/adaptive support logic
- optimizer, learning rate, rollout horizon, seeds, update count
- stochastic PPO action/log-prob contract

Only the treatment adds coordinate-aware raw-logit tail regularization.

## Per-action tail budget

Budgets are calibrated without using the residual lane.

Source:
- frozen robust u50 actor
- existing fixed probe corpus
- T/A/O/S/C preferences

For each action coordinate j:

    tau_j = q95( |z_j| ; robust u50 fixed probes )

No FL-hip-specific threshold or residual-lane statistic is used.

## Treatment loss

For current actor mean logits z:

    L_tail = mean_j [ relu(|z_j| - tau_j)^2 ]

PPO loss is unchanged.

To prevent regularizer gradient hijack, tail contribution is gradient-budgeted:

    alpha = min(1, rho * ||g_ppo|| / (||g_tail|| + eps))
    rho = 0.25

and the actor update uses:

    g_total = g_ppo + alpha g_tail

Thus the treatment changes only the actor-output tail pressure and bounds its instantaneous gradient norm relative to PPO.

CONTROL:
    rho = 0

TREATMENT:
    rho = 0.25

## Short paired screen

Start both arms from the same u75 actor + repaired wide critic checkpoint.
Run the same short continuation training.

Primary checkpoints:
    u0, u5, u10, u20, u25

## Gates

Authority preservation:
- fixed-probe pairwise preference action separation >= 90% of paired control / start
- simplex-tangent action Jacobian >= 90% of paired control / start

Support margin:
- suite3 / C / lane0 must survive
- FL-hip t4..7 action support must not fall below the causal support margin observed for the surviving c=2.5 intervention; report explicitly, do not hard-code FL hip into training loss

Global robustness:
- semantic suites 2-3 minimum survival >= 0.95
- total failed lanes materially lower than paired control

Mechanism:
- coordinate-wise tail excess above tau decreases
- no requirement for zero saturation
- no generic action-norm target

Critic / foundation:
- wide critic path unchanged between arms
- support/head refresh logic identical

Decision:
- survival up + authority retained => repair candidate passes screen
- survival up + authority lost => FAIL
- tails down without survival recovery => mechanism insufficient
- AI-C2/H2 remain blocked until this paired repair passes
