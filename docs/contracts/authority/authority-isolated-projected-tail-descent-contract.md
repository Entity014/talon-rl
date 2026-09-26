# Authority-Isolated Projected + Tail-Descent Contract

Status: **FROZEN BEFORE EXECUTION**
Date: 2026-09-25

The pure conflict projection preserved/grows authority but only enforces first-order non-increase of tail loss; fixed-probe tail still grew 1.41x and robustness did not pass.

Treatment:

1. Remove only the PPO gradient component that conflicts with tail reduction:

    g_proj = g_ppo - min(dot(g_ppo,g_tail)/(||g_tail||^2+eps), 0) g_tail

2. Add a small active tail-descent margin:

    g = g_proj + kappa * ||g_proj||/(||g_tail||+eps) * g_tail

with frozen:

    kappa = 0.05

Thus tail descent has 5% of the projected PPO gradient norm after conflict removal.

All architecture, critic, support, GAE lambda=.95, tau_j, optimizer, preference schedule, and PPO contracts remain unchanged.

Initial screen: 3 updates, seed 73001.

Gates:
- pairwise authority retention >= .90
- tangent Jacobian retention >= .90
- fixed-probe tail loss/fraction improve versus update 0
- residual FL-hip t4..7 support retained
- suites2/3 failures fewer than frozen u75 baseline (9), with min survival trending toward >=.95
